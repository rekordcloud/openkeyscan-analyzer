import os
print(f"PROFILE_PERFORMANCE = '{os.environ.get('PROFILE_PERFORMANCE', 'NOT SET')}'")
print(f"Check: {os.environ.get('PROFILE_PERFORMANCE', '0') == '1'}")
