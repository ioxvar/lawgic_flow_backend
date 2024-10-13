import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import requests
from dotenv import load_dotenv
import json
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS, cross_origin
import re
import anthropic
from functools import lru_cache
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
socketio = SocketIO(app, cors_allowed_origins="*")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not PERPLEXITY_API_KEY or not ANTHROPIC_API_KEY:
    raise ValueError("API keys not set. Please check your .env file.")

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
@cross_origin()
def analyze_case():
    try:
        case_data = request.json
        app.logger.debug(f"Received case data: {case_data}")
        
        if not case_data or 'law_area' not in case_data or 'description' not in case_data:
            return jsonify({"error": "Invalid input. Please provide 'law_area' and 'description'."}), 400
        
        analysis = call_claude_api(case_data)
        citations = call_perplexity_api(case_data)
        
        if "error" in analysis:
            return jsonify(analysis), 500
        if "error" in citations:
            return jsonify(citations), 500
        
        # Merge citations into analysis
        for event in analysis['case_description']['case_timeline']:
            matching_citations = next((item['citations'] for item in citations if item['event'] == event['event']), [])
            event['relevant_case_citations'] = matching_citations
        
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error in analyze_case: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

def extract_json_from_string(s):
    match = re.search(r'\{[\s\S]*\}', s)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def confidence_to_float(confidence):
    confidence_map = {
        "Low": 0.3,
        "Moderate": 0.5,
        "High": 0.8,
        "Very High": 0.9
    }
    if isinstance(confidence, str):
        # Remove any trailing percentage sign and whitespace
        confidence = confidence.rstrip('%').strip()
        # Check if it's a word-based confidence level
        if confidence in confidence_map:
            return confidence_map[confidence]
        # If it's a numeric string, convert to float
        try:
            return float(confidence) / 100
        except ValueError:
            return 0.5  # Default to moderate confidence if unknown
    # If it's already a number, just divide by 100
    return float(confidence) / 100

@lru_cache(maxsize=1)
def get_system_message():
    return """
    You are a legal expert assistant. Analyze the given legal case and provide a detailed response including:
    1. A timeline of key events (at least 5 events)
    2. Applicable laws for each event
    3. Entities involved and their relationships
    4. A preliminary judgment based on the information provided
    5. Potential future outcomes
    6. Confidence level in the analysis (considering completeness of information)
    7. Suggestions for additional information needed for a more accurate analysis
    8. Potential challenges or counterarguments to the preliminary judgment
    9. Estimated duration of the case based on similar historical cases
    10. Jurisdiction analysis and its potential impact on the case
    11. Analysis of any provided patent details or prior art
    Format your response as JSON and make sure it is a valid JSON.
    """

def call_claude_api(case_data):
    user_message = f"""
    Analyze the following legal case:
    Law Area: {case_data['law_area']}
    Case Description: {case_data['description']}
    Additional Details: {case_data.get('additional_details', 'No additional details provided')}
    """
    
    try:
        response = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.2,
            system=get_system_message(),
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        content = response.content[0].text
        parsed_content = extract_json_from_string(content)
        
        if parsed_content is None:
            raise ValueError("Failed to extract valid JSON from Claude's response")

        # Transform the parsed content to match the AnalysisResult interface
        transformed_content = {
            "case_description": {
                "law_area": case_data['law_area'],
                "entities_involved": [{"name": e.get("name", ""), "relationship": e.get("role", "")} for e in parsed_content.get("entities", [])],
                "case_timeline": [{"date": e.get("date", ""), "event": e.get("event", ""), "applicable_laws": e.get("law", "").split(', '), "relevant_case_citations": []} for e in parsed_content.get("timeline", [])]
            },
            "preliminary_judgment": {
                "summary": parsed_content.get("preliminaryJudgment", ""),
                "based_on_information": "",
                "confidence_level": confidence_to_float(parsed_content.get("confidenceLevel", "50%")),
                "potential_outcomes": [{"outcome": o, "probability": 1/len(parsed_content.get("potentialOutcomes", []))} for o in parsed_content.get("potentialOutcomes", [])]
            },
            "similar_historical_cases": [],
            "estimated_duration": {
                "range": parsed_content.get("estimatedDuration", ""),
                "basis": "",
                "based_on_similar_historical_cases": ""
            },
            "jurisdiction_analysis": {
                "jurisdiction": "",
                "impact": "",
                "potential_impact": parsed_content.get("jurisdictionAnalysis", "")
            },
            "potential_challenges_counterarguments": [{"challenge": c, "counterargument": ""} for c in parsed_content.get("potentialChallenges", [])],
            "additional_information_needed": parsed_content.get("additionalInformationNeeded", []),
            "potential_future_outcomes": [{"outcome": o, "confidence_level": 0.5, "additional_information_needed": ""} for o in parsed_content.get("potentialOutcomes", [])]
        }
        
        return transformed_content
    except Exception as e:
        app.logger.error(f"Error calling Claude API: {str(e)}", exc_info=True)
        return {"error": f"Error calling Claude API: {str(e)}"}

def call_perplexity_api(case_data):
    api_url = 'https://api.perplexity.ai/chat/completions'
    system_message = """
    You are a legal expert assistant. For the given legal case, provide only relevant case citations for each key event. Do not provide any analysis or explanation.
    Format your response as a JSON array of objects, where each object contains an 'event' key and a 'citations' key (an array of citation strings).
    """
    user_message = f"""
    Provide relevant case citations for the following legal case:
    Law Area: {case_data['law_area']}
    Case Description: {case_data['description']}
    Additional Details: {case_data.get('additional_details', 'No additional details provided')}
    """
    
    try:
        headers = {
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "llama-3.1-sonar-small-128k-chat",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "max_tokens": 2000
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        # Remove code block markers if present
        content = re.sub(r'```json\n|\n```', '', content)
        
        parsed_content = json.loads(content)
        
        # Transform the parsed content to match the expected format
        transformed_content = []
        for item in parsed_content:
            transformed_item = {
                "event": item.get("event", ""),
                "citations": item.get("citations", [])
            }
            transformed_content.append(transformed_item)
        
        return transformed_content
    except Exception as e:
        app.logger.error(f"Error calling Perplexity API: {str(e)}", exc_info=True)
        return {"error": f"Error calling Perplexity API: {str(e)}"}

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))