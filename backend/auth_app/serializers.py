from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Organization, UserProfile


class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ['id', 'name', 'slug', 'subscription_tier', 'is_active', 'created_at', 'settings']
        read_only_fields = ['id', 'created_at']


class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    first_name = serializers.CharField(source='user.first_name', read_only=True)
    last_name = serializers.CharField(source='user.last_name', read_only=True)
    organization_name = serializers.CharField(source='organization.name', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 
                  'organization', 'organization_name', 'role', 'avatar_url', 
                  'phone', 'preferences', 'created_at', 'last_active', 'is_active']
        read_only_fields = ['id', 'created_at', 'last_active']


class UserRegistrationSerializer(serializers.Serializer):
    """For new organization signup"""
    # Organization
    organization_name = serializers.CharField(max_length=255)
    
    # User
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)
    first_name = serializers.CharField(max_length=150)
    last_name = serializers.CharField(max_length=150)
    
    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already registered")
        return value
    
    def create(self, validated_data):
        from django.utils.text import slugify
        import uuid
        
        # Create organization
        org_name = validated_data['organization_name']
        org_slug = slugify(org_name) + '-' + str(uuid.uuid4())[:8]
        
        organization = Organization.objects.create(
            name=org_name,
            slug=org_slug,
            subscription_tier='free'
        )
        
        # Create user
        user = User.objects.create_user(
            username=validated_data['email'],  # Use email as username
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name']
        )
        
        # Create profile
        profile = UserProfile.objects.create(
            user=user,
            organization=organization,
            role='admin'  # First user is admin
        )
        
        return {
            'user': user,
            'profile': profile,
            'organization': organization
        }


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'profile']
