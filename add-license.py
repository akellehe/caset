#!/usr/bin/env python3
import pathlib

license_cpp = b"""\
// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

"""

license_py = license_cpp.replace(b"//", b"#", 1).replace(b"\n//", b"\n#")

EXTS_CPP = {".cpp", ".cc", ".cxx", ".h", ".hpp"}
EXTS_PY = {".py"}

def prepend_license(path: pathlib.Path):
    if 'add-license' in str(path):
        return

    with open(str(path), 'rb') as fp:
        text = fp.read()

    if b"MIT License" in text.split(b"\n", 5)[0:5]:
        print(str(path), "already has a license")
        return  # already has a license header

    if path.suffix in EXTS_CPP:
        header = license_cpp
    elif path.suffix in EXTS_PY:
        header = license_py
    else:
        return

    print(f"Adding license to {path}")
    with open(str(path), 'wb') as fp:
        fp.write(header + text)
    print(f"Added license to {path}")

if __name__ == "__main__":
    for p in pathlib.Path(".").rglob("*"):
        if p.is_file():
            prepend_license(p)
