import base64, datetime, hashlib, hmac, requests, xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "quratingscoressa")
CONTAINER = os.getenv("AZURE_CONTAINER", "quratingfiltered")
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_KEY")

if not ACCOUNT_KEY:
    raise ValueError("Please set AZURE_STORAGE_KEY environment variable")

KEY_BYTES = base64.b64decode(ACCOUNT_KEY)
ENDPOINT = f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}"

def sign(params):
    xms_date = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    headers = {'x-ms-date': xms_date, 'x-ms-version': '2021-12-02'}
    canon_headers = ''.join(f"{k}:{headers[k]}\n" for k in sorted(headers))
    canon_resource = f"/{ACCOUNT}/{CONTAINER}"
    if params:
        canon_resource += '\n' + '\n'.join(f"{k}:{params[k]}" for k in sorted(params))
    parts = ['GET', '', '', '', '', '', '', '', '', '', '', '']
    string_to_sign = '\n'.join(parts) + '\n' + canon_headers + canon_resource
    signature = base64.b64encode(hmac.new(KEY_BYTES, string_to_sign.encode(), hashlib.sha256).digest()).decode()
    headers['Authorization'] = f"SharedKey {ACCOUNT}:{signature}"
    return headers

total_bytes = 0
blob_count = 0
marker = None
while True:
    params = {'comp': 'list', 'restype': 'container'}
    if marker:
        params['marker'] = marker
    resp = requests.get(ENDPOINT, params=params, headers=sign(params))
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    blobs = root.find('Blobs') or []
    for blob in blobs.findall('Blob'):
        total_bytes += int(blob.find('Properties').find('Content-Length').text)
        blob_count += 1
    marker_el = root.find('NextMarker')
    if not (marker_el is not None and marker_el.text):
        break
    marker = marker_el.text

print(f"blob_count={blob_count} total_bytes={total_bytes} total_gib={total_bytes / 1024 ** 3:.2f}")
