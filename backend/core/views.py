from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.conf import settings
import os
import time

class ConfigureDatabaseView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        db_name = request.data.get('db_name')
        db_host = request.data.get('db_host')
        db_port = request.data.get('db_port')
        db_user = request.data.get('db_user')
        db_password = request.data.get('db_password')

        if not db_name:
            return Response({'error': 'Database Name is required'}, status=400)

        # Path to .env file
        env_path = os.path.join(settings.BASE_DIR, '.env')
        
        # Read existing .env
        env_vars = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        # Update values
        env_vars['DB_NAME'] = db_name
        env_vars['DB_HOST'] = db_host or '127.0.0.1'
        env_vars['DB_PORT'] = db_port or '3306'
        env_vars['DB_USER'] = db_user or 'root'
        if db_password is not None:
            env_vars['DB_PASSWORD'] = db_password
            
        # Write back
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f'{key}={value}\n')
        
        # Trigger reload by touching settings.py
        settings_path = os.path.join(settings.BASE_DIR, 'config', 'settings.py')
        os.utime(settings_path, None)
        
        return Response({'message': 'Database configuration updated. Server restarting...'})
