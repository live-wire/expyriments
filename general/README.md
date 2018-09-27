# Python eggs :egg:

This section covers some basic and advanced topics.
---

- `main.py` includes sugar for:
	- Decorators
	- Thread pool
	- Argumentparser
- `threads.py` and `processes.py` contain sample examples for both.

---

- `pip install httpie`
Usage: 
	- `http <url>` to GET
	- http http://127.0.0.1:8000/snippets/ Accept:application/json
	- http --json POST http://127.0.0.1:8000/snippets/ code="print 456"
	- http -a admin:password123 POST http://127.0.0.1:8000/snippets/ code="print 789"
	- http POST 127.0.0.1:8000/auth/login username=admin password=superuser (BODY DATA)
	- http GET http://localhost:8000/snippets/ Authorization:"Token ca7388cd7f2ca728d9134aa27030d46" (HEADER)

