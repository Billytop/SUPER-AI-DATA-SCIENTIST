import os
import django
from decimal import Decimal
from datetime import datetime, timedelta
import random

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Business
from partners.models import Contact
from inventory.models import Product, Variation, Brand, Category
from sales.models import Transaction, TransactionSellLine

print("Creating simplified sample data...")

# Use existing admin user
admin = User.objects.get(username='admin')
print(f"[OK] Using existing user: {admin.username}")

# Get or create a business for admin
if Business.objects.filter(owner=admin).exists():
    business = Business.objects.filter(owner=admin).first()
    print(f"[OK] Using existing business: {business.name}")
else:
    print("[INFO] No business found. Please create a business first via admin panel.")
    print("[INFO] Or run: python manage.py shell")
    print("[INFO] Then: from core.models import Business; from django.contrib.auth.models import User")
    print("[INFO] Then: Business.objects.create(name='My Business', owner=User.objects.get(username='admin'))")
    exit(0)

# Create a contact
contact, _ = Contact.objects.get_or_create(
    business=business,
    name="Demo Customer",
    defaults={'type': 'customer', 'mobile': '1234567890'}
)
print(f"[OK] Contact: {contact.name}")

# Create products
brand, _ = Brand.objects.get_or_create(business=business, name="Generic Brand")
category, _ = Category.objects.get_or_create(business=business, name="General")

product_names = ['Laptop', 'Phone', 'Tablet', 'Mouse', 'Keyboard']
products = []

for i, pname in enumerate(product_names):
    product, created = Product.objects.get_or_create(
        business=business,
        name=pname,
        defaults={'type': 'single', 'brand': brand, 'category': category, 'sku': f'PROD-{i+1:03d}'}
    )
    
    variation, _ = Variation.objects.get_or_create(
        product=product,
        defaults={'name': 'Standard', 'default_sell_price': Decimal(str(500 + i * 100))}
    )
    
    products.append((product, variation))
    if created:
        print(f"[OK] Product: {pname}")

#Create sample transactions
print("\n[INFO] Creating sample transactions...")
transaction_count = 0

# Create transactions for 2024 and 2025
for year in [2024, 2025]:
    for month in range(1, 13):  # All 12 months
        # Create 3-7 transactions per month
        for _ in range(random.randint(3, 7)):
            day = random.randint(1, 28)
            transaction_date = datetime(year, month, day, 10, 0, 0)
            
            # Random total between 500-5000
            total = Decimal(str(random.randint(500, 5000)))
            
            transaction = Transaction.objects.create(
                business=business,
                type='sell',
                status='final',
                payment_status='paid',
                contact=contact,
                invoice_no=f'INV-{year}-{transaction_count:04d}',
                transaction_date=transaction_date,
                total_before_tax=total,
                tax_amount=Decimal('0'),
                discount_amount=Decimal('0'),
                shipping_charges=Decimal('0'),
                final_total=total,
                created_by=admin
            )
            
            # Add 1-3 line items
            for _ in range(random.randint(1, 3)):
                product, variation = random.choice(products)
                quantity = Decimal(str(random.randint(1, 5)))
                unit_price = Decimal(str(random.randint(100, 1000)))
                
                TransactionSellLine.objects.create(
                    transaction=transaction,
                    product=product,
                    variation=variation,
                    quantity=quantity,
                    unit_price_before_discount=unit_price,
                    unit_price=unit_price,
                    line_discount_amount=Decimal('0'),
                    item_tax=Decimal('0'),
                    unit_price_inc_tax=unit_price
                )
            
            transaction_count += 1

print(f"\n[OK] Created {transaction_count} transactions")
print(f"[OK] Sample data complete!")
print(f"\n{'='*60}")
print(f"NOW TRY THESE QUERIES IN THE CLI CHAT:")
print(f"{'='*60}")
print(f"  mfanyakazi bora ni nani kwa mauzo")
print(f"  total sales 2025")
print(f"  bidhaa bora kwa mauzo")
print(f"  mauzo ya mwaka jana")
