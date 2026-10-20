"""serve/test_literal_think_guard_responses.py - does the #1814 guard reach the Responses dialect?

#2731's Mac evidence (issue comment 2026-10-19) hits this bug through POST /v1/responses from Codex CLI:
the turn came back as a reasoning item alone - no message, no call - and the client was told nothing about
why.  The guard lives in Service.run, which every dialect shares, but the flag it reports
(`ended_inside_thinking`) was only ever wired into the chat-completions reply.  Measured here with a mock
engine (no GPU, no pack): does the continuation fire on the Responses API, and does the report reach it?

    python -m unittest serve.test_literal_think_guard_responses -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, THINK_END  # noqa: E402
from serve.server import (LITERAL_MARKER_WRITES, LITERAL_THINK_RETRIES, REASONING_CLOSE,  # noqa: E402
                          ByteTokenizer, Service, serve)
from serve.test_literal_think_guard import MARKER, NO_ANSWER, ROOT, SpecimenEngine  # noqa: E402

LT, GT = chr(0x3C), chr(0x3E)
# The template's own call shape (see serve/test_server.py's tool-call cases), built from code points so
# nothing in transit turns these tags into special tokens or drops them.
CALL = (LT + "tool_call" + GT + "\n" + LT + "function=exec_command" + GT + "\n"
        + LT + "parameter=cmd" + GT + "\nls ~/Library/Logs/strata | head -3\n"
        + LT + "/parameter" + GT + "\n" + LT + "/tool_call" + GT)
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a shell command.",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]


class TagThenCallEngine(SpecimenEngine):
    """#2731's Mac shape: the thought is writing the closing tag between backticks and the turn-end token
    arrives mid-sentence - no answer, no call.  A prompt that ends with the written-back text (#1178) is the
    thought going on: it finishes the sentence, closes its own thinking, and makes the call it promised."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.tok.decode(ids)
        if text.endswith(MARKER):        # the written-back text: the thought goes on and makes the call
            out = self.tok.encode(" is written between backticks here. It closes the thinking block."
                                  + THINK_END + "\n\n" + CALL)
        elif text.endswith(REASONING_CLOSE):     # the close instead of the write-back: still owes the call
            out = self.tok.encode("\n\n" + CALL)
        else:
            out = self.tok.encode(self.raw)
        for t in (out + self.tok.encode(MARKER, parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


def _loop_threads() -> set:
    return {t for t in threading.enumerate() if t.name.endswith("(_loop)")}


class ResponsesDialect(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self._closed = True

    def tearDown(self):
        self.close()

    def boot(self, engine_cls, raw="", guard_on=True):
        self.close()
        self.engine = engine_cls(self.tok, raw)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.literal_think_guard = guard_on
        self._loops_before = _loop_threads()
        self._closed = False
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        httpd = getattr(self, "httpd", None)
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        deadline = time.time() + 5.0
        while time.time() < deadline and (_loop_threads() - self._loops_before):
            time.sleep(0.02)

    def post(self, body, path="/v1/responses"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as log:
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse()
                parsed = json.loads(r.read().decode())
            self.log = log.getvalue()
            if r.status >= 400:
                print(f"[measured] HTTP {r.status}: {json.dumps(parsed)[:400]}")
            return r.status, parsed
        finally:
            c.close()

    def responses_request(self, last="Write the end tag in your thinking, then run the command."):
        return {"model": "m", "store": False, "tool_choice": "auto", "tools": TOOLS,
                "instructions": "You are a coding agent. Use the exec_command tool.",
                "input": [{"role": "user", "content": last}]}

    @staticmethod
    def items_of(r: dict) -> list:
        return [i["type"] for i in r.get("output", [])]

    @staticmethod
    def text_of(r: dict) -> str:
        out = []
        for item in r.get("output", []):
            for part in item.get("content", []) or []:
                out.append(part.get("text", ""))
        return "".join(out)

    # ------------------------------------------------------------------ the continuation

    def test_the_guard_continues_a_responses_turn_that_end_inside_the_thinking(self):
        """Guard on: the turn the Mac saw as a reasoning-only item gets an answer, and the call."""
        self.boot(SpecimenEngine, NO_ANSWER[0]["raw"])
        code, r = self.post(self.responses_request())
        self.assertEqual(code, 200)
        print("\n[measured] guard ON /v1/responses items:", self.items_of(r),
              "| status:", r["status"], "| prompts:", len(self.engine.prompts))
        print("[measured] answer text:", self.text_of(r)[:90])
        print("[measured] keys of the response object:", sorted(r.keys()))
        self.assertEqual(len(self.engine.prompts), 2)          # the reply, then the continuation
        self.assertIn("message", self.items_of(r))             # not a reasoning-only item
        self.assertIn(SpecimenEngine.ANSWER, self.text_of(r))
        self.assertEqual(r["status"], "completed")

    def test_the_tool_call_the_lost_turn_promised_survives(self):
        """#2731's Mac repro ends on a backtick mid-sentence and then owes the client a function_call."""
        raw = "The Qwen thinking block end tag is "
        self.boot(TagThenCallEngine, raw)
        code, r = self.post(self.responses_request())
        self.assertEqual(code, 200)
        print("\n[measured] tag-then-call items:", self.items_of(r),
              "| prompts:", len(self.engine.prompts))
        print("[measured] reasoning tail:", self.text_of(r)[:120])
        print("[measured] log:\n" + "".join("    " + l for l in self.log.splitlines() if "thinking" in l))
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertIn("function_call", self.items_of(r))       # the call Codex never got
        call = [i for i in r["output"] if i["type"] == "function_call"][0]
        print("[measured] function_call name:", call["name"], "| status:", call.get("status"),
              "| arguments repr:", repr(call["arguments"]))
        self.assertEqual(call["name"], "exec_command")
        self.assertIn("ls", call["arguments"])

    # ------------------------------------------------------------------ the report

    def test_the_chat_dialect_reports_the_empty_reply_and_the_responses_dialect_does_not(self):
        """The cap case: a reply nothing can save.  The flag is on the chat reply; is it on the Responses one?"""
        from serve.test_literal_think_guard import AlwaysStopsEngine   # every pass ends inside the thinking
        self.boot(AlwaysStopsEngine, NO_ANSWER[0]["raw"])
        _, chat = self.post({"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                             "max_tokens": 4000}, path="/v1/chat/completions")
        chat_flag = "ended_inside_thinking" in chat
        self.boot(AlwaysStopsEngine, NO_ANSWER[0]["raw"])         # a fresh service for the second dialect
        code, resp = self.post(self.responses_request())
        resp_flag = "ended_inside_thinking" in json.dumps(resp)
        print("\n[measured] chat dialect reports it:", chat_flag, "| keys:", sorted(chat.keys()))
        print("[measured] responses dialect reports it:", resp_flag,
              "| keys:", sorted(resp.keys()), "| items:", self.items_of(resp),
              "| status:", resp["status"], "| usage:", resp.get("usage"))
        print("[measured] totals:", {k: v for k, v in self.svc.totals.items() if "thinking" in k})
        self.assertTrue(chat_flag)                               # what the PR promises for chat
        self.assertNotIn("ended_inside_thinking", resp)          # what the Responses object carries

    def test_the_guard_off_responses_turn_is_a_reasoning_item_alone(self):
        """What the Mac client saw: guard off, no answer, and nothing in the body that says why."""
        self.boot(SpecimenEngine, NO_ANSWER[0]["raw"], guard_on=False)
        code, r = self.post(self.responses_request())
        print("\n[measured] guard OFF items:", self.items_of(r), "| status:", r["status"],
              "| keys:", sorted(r.keys()))
        print("[measured] reasoning tail:", self.text_of(r)[-60:])
        self.assertEqual(code, 200)
        self.assertEqual(self.items_of(r), ["reasoning"])        # no message item, no call
        self.assertNotIn("ended_inside_thinking", r)
        self.assertEqual(len(self.engine.prompts), 1)


if __name__ == "__main__":
    unittest.main()
