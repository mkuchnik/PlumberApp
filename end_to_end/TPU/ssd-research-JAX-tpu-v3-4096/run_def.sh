list_python_programs() {
	ps aux | grep python | grep -v "grep python" | awk '{print $2}'
}

kill_python_programs() {
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
}
