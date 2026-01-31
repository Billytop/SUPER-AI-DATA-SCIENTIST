from django.contrib import admin
from .models import Organization, UserProfile


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'subscription_tier', 'is_active', 'created_at')
    list_filter = ('subscription_tier', 'is_active', 'created_at')
    search_fields = ('name', 'slug')
    readonly_fields = ('id', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'name', 'slug', 'is_active')
        }),
        ('Subscription', {
            'fields': ('subscription_tier', 'max_users', 'max_conversations')
        }),
        ('Settings', {
            'fields': ('settings',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'organization', 'role', 'is_active', 'last_active')
    list_filter = ('role', 'is_active', 'organization')
    search_fields = ('user__username', 'user__email', 'user__first_name', 'user__last_name')
    readonly_fields = ('id', 'created_at', 'last_active')
    
    fieldsets = (
        ('User', {
            'fields': ('id', 'user', 'organization')
        }),
        ('Profile', {
            'fields': ('role', 'avatar_url', 'phone', 'is_active')
        }),
        ('Preferences', {
            'fields': ('preferences',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'last_active')
        }),
    )
