from django.db import models
from core.models import Business
from partners.models import Contact
from django.contrib.auth.models import User

class Project(models.Model):
    """
    Project Management logic.
    """
    STATUSES = [('not_started', 'Not Started'), ('in_progress', 'In Progress'), ('on_hold', 'On Hold'), ('completed', 'Completed')]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=50, choices=STATUSES, default='not_started')
    
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)
    
    lead = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='lead_projects')
    members = models.ManyToManyField(User, related_name='projects')
    contact = models.ForeignKey(Contact, on_delete=models.SET_NULL, blank=True, null=True, related_name='projects')
    
    description = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

class ToDo(models.Model):
    """
    Essentials/ToDo logic.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    task = models.TextField()
    is_completed = models.BooleanField(default=False)
    due_date = models.DateField(blank=True, null=True)
    priority = models.CharField(max_length=20, choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')])
    
    created_at = models.DateTimeField(auto_now_add=True)

class Campaign(models.Model):
    """
    CRM Campaign logic.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=50, choices=[('email', 'Email'), ('sms', 'SMS')])
    
    subject = models.CharField(max_length=255, blank=True, null=True)
    body = models.TextField()
    
    sent_on = models.DateTimeField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
