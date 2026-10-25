"""Local TypeSafe-compliant Decision Model Server on port 9931.

Bridges TypeSafeClient /v1/systemone requests directly to Ollama's clef-flash runner.
"""

import json
import subprocess
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Local Clef Decision Server")

@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {
                "id": "clef-flash",
                "name": "Clef Flash Decision Model",
                "type": "decision",
                "capabilities": ["choice", "noul", "score"]
            }
        ]
    }

@app.post("/v1/systemone")
async def system_one(request: Request):
    payload = await request.json()
    state = payload.get("state", "")
    questions = payload.get("questions", {})
    model = payload.get("model", "clef-flash")
    
    # Format state string
    if not isinstance(state, str):
        state_str = json.dumps(state)
    else:
        state_str = state

    answers = {}
    total_input_tokens = 0
    total_output_tokens = 0

    # Build Ollama CLI arguments for all questions in one shot
    # ollama run clef-flash --format json --ask "q1_inst" --choice c1="desc" ...
    cmd = ["ollama", "run", model, "--format", "json"]

    q_order = []
    for q_id, q_data in questions.items():
        q_order.append(q_id)
        instructions = q_data.get("instructions", "")
        q_type = q_data.get("type", "choice")
        cmd.extend(["--ask", instructions])

        if q_type == "choice":
            criteria = q_data.get("criteria", {})
            if isinstance(criteria, dict):
                for k, v in criteria.items():
                    desc = v if v else k
                    cmd.extend(["--choice", f"{k}={desc}"])
            elif isinstance(criteria, list):
                for k in criteria:
                    cmd.extend(["--choice", f"{k}={k}"])

    try:
        proc = subprocess.run(
            cmd,
            input=state_str,
            text=True,
            capture_output=True,
            check=True
        )
        stdout = proc.stdout.strip()
        # Find JSON object in stdout (ignoring spinner chars)
        json_start = stdout.find("{")
        json_end = stdout.rfind("}")
        if json_start != -1 and json_end != -1:
            raw_json = stdout[json_start:json_end+1]
            data = json.loads(raw_json)
            
            raw_answers = data.get("answers", {})
            # Remap q1, q2... to actual question IDs if needed
            remapped_answers = {}
            for idx, q_id in enumerate(q_order):
                default_key = f"q{idx+1}"
                if q_id in raw_answers:
                    remapped_answers[q_id] = raw_answers[q_id]
                elif default_key in raw_answers:
                    remapped_answers[q_id] = raw_answers[default_key]
                elif str(idx) in raw_answers:
                    remapped_answers[q_id] = raw_answers[str(idx)]
                elif len(raw_answers) == 1 and idx == 0:
                    first_val = next(iter(raw_answers.values()))
                    remapped_answers[q_id] = first_val

            return JSONResponse({
                "model": model,
                "answers": remapped_answers,
                "usage": data.get("usage", {"input_tokens": 0, "output_tokens": 0})
            })
    except Exception as e:
        # Fallback to single question query if batched fails
        for q_id, q_data in questions.items():
            instructions = q_data.get("instructions", "")
            q_type = q_data.get("type", "choice")
            single_cmd = ["ollama", "run", model, "--format", "json", "--ask", instructions]
            criteria = q_data.get("criteria", {})
            if q_type == "choice":
                if isinstance(criteria, dict):
                    for k, v in criteria.items():
                        desc = v if v else k
                        single_cmd.extend(["--choice", f"{k}={desc}"])
                elif isinstance(criteria, list):
                    for k in criteria:
                        single_cmd.extend(["--choice", f"{k}={k}"])
            try:
                sub_proc = subprocess.run(
                    single_cmd,
                    input=state_str,
                    text=True,
                    capture_output=True,
                    check=True
                )
                sub_out = sub_proc.stdout.strip()
                j_s = sub_out.find("{")
                j_e = sub_out.rfind("}")
                if j_s != -1 and j_e != -1:
                    sub_data = json.loads(sub_out[j_s:j_e+1])
                    sub_ans = sub_data.get("answers", {})
                    ans_val = next(iter(sub_ans.values())) if sub_ans else None
                    if ans_val:
                        answers[q_id] = ans_val
                    u = sub_data.get("usage", {})
                    total_input_tokens += u.get("input_tokens", 0)
                    total_output_tokens += u.get("output_tokens", 0)
            except Exception as sub_e:
                print(f"Error querying {q_id}: {sub_e}")

        return JSONResponse({
            "model": model,
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }
        })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9931)
