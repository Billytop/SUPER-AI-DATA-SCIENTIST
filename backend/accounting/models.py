from django.db import models
from core.models import Business
from django.contrib.auth.models import User
from sales.models import Transaction, TransactionPayment

class AccountType(models.Model):
    """
    Mirrors 'account_types' table.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    parent_account = models.ForeignKey('self', on_delete=models.CASCADE, blank=True, null=True, related_name='sub_types')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Account(models.Model):
    """
    Mirrors 'accounts' table. Chart of Accounts.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='accounts')
    name = models.CharField(max_length=255)
    account_number = models.CharField(max_length=255)
    account_type = models.ForeignKey(AccountType, on_delete=models.CASCADE, blank=True, null=True)
    note = models.TextField(blank=True, null=True)
    is_closed = models.BooleanField(default=False)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.account_number})"

class AccountTransaction(models.Model):
    """
    Mirrors 'account_transactions' table. The Immutable Ledger.
    """
    TYPES = [('debit', 'Debit'), ('credit', 'Credit')]
    SUB_TYPES = [('opening_balance', 'Opening Balance'), ('fund_transfer', 'Fund Transfer'), ('deposit', 'Deposit')]

    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='transactions')
    type = models.CharField(max_length=50, choices=TYPES)
    sub_type = models.CharField(max_length=50, choices=SUB_TYPES, blank=True, null=True)
    amount = models.DecimalField(max_digits=22, decimal_places=4)
    reff_no = models.CharField(max_length=255, blank=True, null=True)
    operation_date = models.DateTimeField()
    
    # Linking to Sales
    transaction = models.ForeignKey(Transaction, on_delete=models.SET_NULL, blank=True, null=True)
    transaction_payment = models.ForeignKey(TransactionPayment, on_delete=models.SET_NULL, blank=True, null=True)
    transfer_transaction = models.ForeignKey('self', on_delete=models.SET_NULL, blank=True, null=True)
    
    note = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.type.upper()} {self.amount} - {self.account.name}"
