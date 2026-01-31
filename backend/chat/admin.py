from django.contrib import admin
from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    fields = ('role', 'content', 'created_at')
    readonly_fields = ('created_at',)
    can_delete = False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'organization', 'message_count', 'is_archived', 'updated_at')
    list_filter = ('is_archived', 'is_pinned', 'created_at', 'organization')
    search_fields = ('title', 'user__username', 'user__email')
    readonly_fields = ('id', 'created_at', 'updated_at', 'message_count')
    inlines = [MessageInline]
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'title', 'user', 'organization')
        }),
        ('Status', {
            'fields': ('is_archived', 'is_pinned')
        }),
        ('Metadata', {
            'fields': ('metadata',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'message_count')
        }),
    )


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content_preview', 'intent', 'created_at')
    list_filter = ('role', 'intent', 'created_at')
    search_fields = ('content', 'conversation__title')
    readonly_fields = ('id', 'created_at')
    
    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'
    
    fieldsets = (
        ('Message', {
            'fields': ('id', 'conversation', 'role', 'content')
        }),
        ('AI Metadata', {
            'fields': ('intent', 'sql_query', 'processing_time', 'tokens')
        }),
        ('Attachments', {
            'fields': ('chart_path', 'export_paths')
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )
