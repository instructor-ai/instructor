"""
partialjson - Parse Partial and incomplete JSON in python
Copyright (c) 2023 Nima Akbarzadeh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json


class JSONParser:
    def __init__(self):
        self.parsers = {
            " ": self.parse_space,
            "\r": self.parse_space,
            "\n": self.parse_space,
            "\t": self.parse_space,
            "[": self.parse_array,
            "{": self.parse_object,
            '"': self.parse_string,
            "t": self.parse_true,
            "f": self.parse_false,
            "n": self.parse_null,
        }
        # Adding parsers for numbers
        for c in "0123456789.-":
            self.parsers[c] = self.parse_number

        self.last_parse_reminding = None
        self.on_extra_token = self.default_on_extra_token

    def default_on_extra_token(self, text, data, reminding):
        pass

    def parse(self, s):
        if len(s) >= 1:
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                data, reminding = self.parse_any(s, e)
                self.last_parse_reminding = reminding
                if self.on_extra_token and reminding:
                    self.on_extra_token(s, data, reminding)
                return json.loads(json.dumps(data))
        else:
            return json.loads("{}")

    def parse_any(self, s, e):
        if not s:
            raise e
        parser = self.parsers.get(s[0])
        if not parser:
            raise e
        return parser(s, e)

    def parse_space(self, s, e):
        return self.parse_any(s.strip(), e)

    def parse_array(self, s, e):
        s = s[1:]  # skip starting '['
        acc = []
        s = s.strip()
        while s:
            if s[0] == "]":
                s = s[1:]  # skip ending ']'
                break
            res, s = self.parse_any(s, e)
            acc.append(res)
            s = s.strip()
            if s.startswith(","):
                s = s[1:]
                s = s.strip()
        return acc, s

    def parse_object(self, s, e):
        s = s[1:]  # skip starting '{'
        acc = {}
        s = s.strip()
        while s:
            if s[0] == "}":
                s = s[1:]  # skip ending '}'
                break
            key, s = self.parse_any(s, e)
            s = s.strip()

            # Handle case where object ends after a key
            if not s or s[0] == "}":
                acc[key] = None
                break

            # Expecting a colon after the key
            if s[0] != ":":
                raise e  # or handle this scenario as per your requirement

            s = s[1:]  # skip ':'
            s = s.strip()

            # Handle case where value is missing or incomplete
            if not s or s[0] in ",}":
                acc[key] = None
                if s.startswith(","):
                    s = s[1:]
                break

            value, s = self.parse_any(s, e)
            acc[key] = value
            s = s.strip()
            if s.startswith(","):
                s = s[1:]
                s = s.strip()
        return acc, s

    def parse_string(self, s, e):  # noqa: ARG002
        end = s.find('"', 1)
        while end != -1 and s[end - 1] == "\\":  # Handle escaped quotes
            end = s.find('"', end + 1)
        if end == -1:
            # Return the incomplete string without the opening quote
            return s[1:], ""
        str_val = s[: end + 1]
        s = s[end + 1 :]
        return json.loads(str_val), s

    def parse_number(self, s, e):
        i = 0
        while i < len(s) and s[i] in "0123456789.-":
            i += 1
        num_str = s[:i]
        s = s[i:]
        if not num_str or num_str.endswith(".") or num_str.endswith("-"):
            return num_str, ""  # Return the incomplete number as is
        try:
            num = (
                float(num_str)
                if "." in num_str or "e" in num_str or "E" in num_str
                else int(num_str)
            )
        except ValueError as e:
            raise e
        return num, s

    def parse_true(self, s, e):
        if s.startswith("true"):
            return True, s[4:]
        raise e

    def parse_false(self, s, e):
        if s.startswith("false"):
            return False, s[5:]
        raise e

    def parse_null(self, s, e):
        if s.startswith("null"):
            return None, s[4:]
        raise e
