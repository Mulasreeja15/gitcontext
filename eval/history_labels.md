# Verification of Requests history labels

These eight questions target real commits within the latest 500 commits of the
indexed Requests checkout (`611c6162`). Labels were chosen from commit messages,
not from retrieval results. Commit messages document intent; they do not guarantee
that the implementation is still unchanged at HEAD. Each commit touches a supported
Python or documentation file. No PR API or token is needed for these labels.

| Question | Expected label and evidence | Why the label answers the question |
|---|---|---|
| Why did Requests clarify the README's bad commit error as a bad commit timestamp? | [commit::8ff173b1](https://github.com/psf/requests/commit/8ff173b186f135670879db9fbe1e68a6b5209c89) | The message identifies vague error wording and explains that naming the bad timestamp helps users understand the cloning issue and the required Git flag. |
| Why was URLRequired removed from the documented Requests exceptions? | [commit::e361622f](https://github.com/psf/requests/commit/e361622fae03f08b16c48a0a1414a718d9d45d25) | It explains that URLRequired was dead code and never raised, while actual invalid-URL exceptions were omitted from the docs. |
| Why did Requests start considering TLS settings when choosing a connection pool? | [commit::c0813a2d](https://github.com/psf/requests/commit/c0813a2d910ea6b4f8438b91d315b8d181302356) | It describes a connection opened with verification disabled potentially being reused for a subsequent request that expected verification. |
| Why was JSONDecodeError's __reduce__ method fixed for pickling? | [commit::3ff3ff21](https://github.com/psf/requests/commit/3ff3ff21dd45957c9e143cd500291959bb15f690) | The message explains that the inheritance order selected the wrong reduction method, losing constructor arguments and breaking deserialization, including in process pools. |
| Why can the total connection timeout exceed the timeout specified by the caller? | [commit::a64f32ba](https://github.com/psf/requests/commit/a64f32ba453bc19aa679018838bee8ef8cc9a68b) | It explains that urllib3 tries multiple IP addresses sequentially, applying the supplied timeout to each attempt. |
| Why was an automatic retry example added to the advanced usage documentation? | [commit::cdbc2e27](https://github.com/psf/requests/commit/cdbc2e271529f467b278b2760f12ee0b5d6930d3) | The message calls retries a common advanced need and explains that the existing HTTPAdapter documentation was buried and insufficiently specific. |
| Why was the Python version upper bound of less than 4 removed from python_requires? | [commit::16a17a3c](https://github.com/psf/requests/commit/16a17a3ca7134b0a56c49653165ef37cb6acfece) | The message explains that speculative future Python upper bounds misuse python_requires and are discouraged by packaging guidance. |
| Why did Requests replace chardet with charset_normalizer as its default dependency? | [commit::2ed84f55](https://github.com/psf/requests/commit/2ed84f55b22f19a1e1e8eea2e50963dce62052d3) | It explains that replacing LGPL with MIT removes downstream license ambiguity when bundling Requests into binaries, while optional chardet support remains an escape hatch. |
