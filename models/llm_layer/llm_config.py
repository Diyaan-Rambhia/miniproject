"""
LLM prompt configuration and grounded attack signatures.

Contains provider config and the known ATTACK_SIGNATURES mapping used to
ground explanation prompts. Imported by the prompt-building module.
"""

# SET YOUR LLM PROVIDER HERE — this is the only line to change
API_PROVIDER = "anthropic"   # "anthropic" or "openai"

API_KEY_ENV_VAR = "LLM_API_KEY"

ATTACK_SIGNATURES = {
    "DoS Hulk": "a high volume of HTTP requests aimed at exhausting server resources",
    "DoS GoldenEye": "a flood of HTTP GET/POST requests designed to overwhelm a web server",
    "DoS Slowloris": "many slow, incomplete HTTP requests that hold connections open to exhaust server threads",
    "DoS Slowhttptest": "a slow-rate denial-of-service technique that keeps connections open with minimal data",
    "PortScan": "sequential connection attempts across many ports, typical of network reconnaissance",
    "FTP-Patator": "repeated FTP login attempts, consistent with a brute-force credential attack",
    "SSH-Patator": "repeated SSH login attempts, consistent with a brute-force credential attack",
    "Bot": "command-and-control style periodic communication, consistent with botnet activity",
    "Web Attack – Brute Force": "repeated login attempts against a web application",
    "Web Attack – XSS": "traffic patterns consistent with a cross-site scripting attempt",
    "Web Attack – SQL Injection": "traffic patterns consistent with a SQL injection attempt",
    "Infiltration": "traffic consistent with an attacker who has already gained internal network access",
    "Heartbleed": "an exploit attempt against the OpenSSL Heartbleed vulnerability",
    "DDoS": "a distributed flood of traffic from many sources aimed at overwhelming a target",
}
