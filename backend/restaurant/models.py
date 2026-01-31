from django.db import models
from core.models import Business
from partners.models import Contact
from django.contrib.auth.models import User

class ResTable(models.Model):
    """
    Restaurant Tables.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Booking(models.Model):
    """
    Table Bookings.
    """
    STATUSES = [('booked', 'Booked'), ('completed', 'Completed'), ('cancelled', 'Cancelled')]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    contact = models.ForeignKey(Contact, on_delete=models.CASCADE)
    table = models.ForeignKey(ResTable, on_delete=models.SET_NULL, blank=True, null=True)
    waiter = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='bookings')
    
    booking_start = models.DateTimeField()
    booking_end = models.DateTimeField()
    status = models.CharField(max_length=50, choices=STATUSES, default='booked')
    
    note = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
