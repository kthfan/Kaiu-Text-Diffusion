# -*- coding: utf-8 -*-
import sys
import os
import http.server
from http.server import BaseHTTPRequestHandler
import mimetypes

ROOT = '../..'
ROOT = './'
PASSWD = None

ServerClass  = http.server.HTTPServer
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = os.path.join(ROOT, self.path[1:])
        if os.path.isdir(path): # if no file specified, index.html assigned. 
            path = os.path.join(path, 'index.html')
            print(path)
        if path == './index.html': # redirect to src/index.html
            self.send_response(302)
            self.send_header('Location', f'/src/index.html')
            self.end_headers()
        elif os.path.isfile(path):
            self.send_response(200)
            with open(path, 'rb') as f: 
                data = f.read()
            self.send_header('content-type', mimetypes.guess_type(path)[0] or '*/*')
            self.send_header('content-length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 80

server_address = ('127.0.0.1', port)

httpd = ServerClass(server_address, MyHandler)

sa = httpd.socket.getsockname()
print("Serving HTTP on", sa[0], "port", sa[1], "...")
httpd.serve_forever()
