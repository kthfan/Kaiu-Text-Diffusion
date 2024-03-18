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
            path = os.path.join(path, 'src/index.html')
        if os.path.isfile(path):
            self.send_response(200)
            with open(path, 'rb') as f: 
                data = f.read()
            self.send_header('content-type', mimetypes.guess_type(path)[0] or '*/*')
            self.send_header('content-length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    # def do_OPTIONS(self):
    #     self.send_response(200, "ok")
    #     self.send_header('Access-Control-Allow-Origin', '*')
    #     self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
    #     self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
    #     self.send_header("Access-Control-Allow-Headers", "Content-Type")
    #     self.end_headers()

if sys.argv[1:]:
    port = int(sys.argv[1])
else:
    port = 80

server_address = ('127.0.0.1', port)

httpd = ServerClass(server_address, MyHandler)

sa = httpd.socket.getsockname()
print("Serving HTTP on", sa[0], "port", sa[1], "...")
httpd.serve_forever()
