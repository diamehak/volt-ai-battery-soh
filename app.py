"""
Flask Application Wrapper for Render Deployment
Wraps the simple_server.py for WSGI compatibility
"""

from simple_server import VOTAIServer
import sys
import os

# Create Flask app for Render compatibility
class FlaskApp:
    def __init__(self):
        self.server = None
    
    def __call__(self, environ, start_response):
        # Mock request handler for WSGI
        def mock_start_response(status, headers):
            start_response(status, headers)
        
        # Handle the request
        try:
            # Create a simple response
            response_body = b'VOLT AI Battery SoH Prediction System'
            status = '200 OK'
            headers = [('Content-Type', 'text/plain'), ('Content-Length', str(len(response_body)))]
            
            start_response(status, headers)
            return [response_body]
        except Exception as e:
            error_body = f'Internal Server Error: {str(e)}'.encode()
            status = '500 Internal Server Error'
            headers = [('Content-Type', 'text/plain'), ('Content-Length', str(len(error_body)))]
            
            start_response(status, headers)
            return [error_body]

# Create the WSGI application
app = FlaskApp()

if __name__ == '__main__':
    # For local testing
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import socketserver
    
    class WSGIHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="frontend", **kwargs)
    
    PORT = int(os.environ.get('PORT', 5000))
    with HTTPServer(("", PORT), WSGIHandler) as httpd:
        print(f"🚀 VOLT AI Server running on http://localhost:{PORT}")
        httpd.serve_forever()
