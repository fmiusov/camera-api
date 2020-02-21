import json
import requests
from requests.auth import HTTPBasicAuth
from requests.auth import HTTPDigestAuth

# "http://admin:jg00dman@192.168.1.114/cgi-bin/mjpg/video.cgi"

camera_1_http = "http://admin:gj00dman@192.168.1.114/cgi-bin/mjpg/video.cgi"
camera_2_http = "http://192.168.1.109/cgi-bin/mjpg/video.cgi"

# r = requests.get(camera_2_http, auth=('admin', 'uwasat0ad'), stream=True)
r = requests.get(camera_1_http, auth=HTTPBasicAuth('admin', 'gj00dman'), stream=True)

print ("camera response:", r)
print (r.encoding)

if r.encoding is None:
    r.encoding = 'utf-8'

for line in r.iter_lines(decode_unicode=True):
    if line:
        print(json.loads(line))