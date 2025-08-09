@socketio.on('start_system')
def handle_start_system():
    global chatty_ai, system_running
    try:
        # These lines are crucial - they update the web interface:
        socketio.emit('status', {'connected': True, 'system_running': False})
        socketio.emit('log', {'message': 'Starting system...'})
        
        chatty_ai = ChattyAI()  # This works fine (proven by debug)
        system_running = True
        
        # THIS IS THE KEY - emit success status:
        socketio.emit('status', {'connected': True, 'system_running': True})
        socketio.emit('log', {'message': 'System started successfully!'})
        
    except Exception as e:
        socketio.emit('log', {'message': f'Error: {str(e)}'})