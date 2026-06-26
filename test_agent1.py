import subprocess
import time
import requests
import sys

from horizon_router.client import HorizonRouter

print("Starting uvicorn server...")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "horizon_router.server:app", "--port", "8000"])
time.sleep(2)

try:
    print("Testing client...")
    r = HorizonRouter("http://localhost:8000")
    
    try:
        assert r.should_delegate(estimated_depth=35, model="gpt-4o") == True
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e.response.text}")
        raise
    print("test 1 passed")
    
    assert r.should_delegate(estimated_depth=8, model="gpt-4o") == False
    print("test 2 passed")
    
    assert r.recommend_model(18) == "qwen-2.5-7b"
    print("test 3 passed")
    
    print("Testing GET /horizons...")
    resp = requests.get("http://localhost:8000/horizons")
    resp.raise_for_status()
    data = resp.json()
    gpt4o_data = next((d for d in data if d["model"] == "gpt-4o"), None)
    assert gpt4o_data is not None
    assert gpt4o_data["d_star"] == 22.0
    print("test 4 passed")
    
    print("All tests passed!")
finally:
    proc.terminate()
    proc.wait()
