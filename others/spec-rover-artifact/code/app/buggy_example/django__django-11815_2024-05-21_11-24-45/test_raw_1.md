To create a standalone Python file `reproducer.py` that reproduces the described issue, we need to simulate the Django environment, including model definition and migration generation. However, it's important to note that directly simulating the Django migration process in a simple script is complex due to Django's reliance on its project structure and settings. Instead, the script will mimic the scenario described and check for the expected behavior. This approach focuses on demonstrating the issue rather than integrating with Django's migration system.

```python
# reproducer.py
import sys
from enum import Enum
from django.utils.translation import gettext_lazy as _
from django.db import models
from django.conf import settings

# Setup minimal Django configuration
settings.configure(
    INSTALLED_APPS=[
        '__main__',  # Treat this script as a Django app for model discovery
    ],
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    USE_I18N=True,
    LANGUAGE_CODE='en-us',
    BASE_DIR='',
)

# Required to finalize Django setup
import django
django.setup()

# Define the Enum and Model as described in the issue
class Status(Enum):
    GOOD = _('Good')  # 'Good' will be translated
    BAD = _('Bad')    # 'Bad' will be translated

    def __str__(self):
        return self.name

class Item(models.Model):
    status = models.CharField(default=Status.GOOD, max_length=128)

# Mimic the migration process and check for the issue
def check_migration_issue():
    # Attempt to simulate the issue described
    try:
        # This is a simplification. In reality, Django migrations would not execute this way.
        # Instead, this simulates the effect of having the Enum value directly in the migration file.
        item = Item(status=Status.GOOD.value)  # Simulate using the Enum value as in the migration
        assert item.status == Status.GOOD.name, "Migration used Enum value instead of name"
    except ValueError as e:
        print(f"AssertionError: {e}")
        sys.exit(1)  # Exit with error code to indicate the issue is present

    print("Issue not present. Exiting with code 0.")
    sys.exit(0)

if __name__ == '__main__':
    check_migration_issue()
```

This script sets up a minimal Django environment, defines an Enum and a model using that Enum, and then simulates the issue described. It attempts to create an instance of the model using the Enum's value directly, as would be the case if the migration incorrectly used the value instead of the name. If the issue is present (i.e., Django migrations are incorrectly handling Enum values), the script will exit with an error code. Otherwise, it exits with code 0, indicating the issue is not present.

Please note, this script assumes you have Django installed in your environment (`pip install Django`) and is a simplified representation meant to demonstrate the issue rather than a direct interaction with Django's migration system.