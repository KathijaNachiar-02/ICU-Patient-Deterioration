
import sqlite3
import hashlib

conn = sqlite3.connect('hospital.db')
cursor = conn.cursor()

# Clear existing table
cursor.execute('DROP TABLE IF EXISTS users')

# Create fresh table
cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        name TEXT NOT NULL,
        department TEXT
    )
''')

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# Add users with passwords
users = [
    ('admin@smartcare.com', hash_pw('admin123'), 'admin', 'Dr. Admin', 'Administration'),
    ('doctor1@smartcare.com', hash_pw('doctor123'), 'doctor', 'Dr. Sarah Wilson', 'Cardiology'),
    ('doctor2@smartcare.com', hash_pw('doctor456'), 'doctor', 'Dr. Raj Patel', 'Neurology'),
    ('nurse@smartcare.com', hash_pw('nurse123'), 'doctor', 'Nurse Emily', 'ICU'),
    ('drsindhya@smartcare.com', hash_pw('sindhya123'), 'doctor', 'Dr. Sindhya', 'Emergency'),
]

cursor.executemany('INSERT INTO users (email, password, role, name, department) VALUES (?, ?, ?, ?, ?)', users)
conn.commit()

print('✅ Database reset successfully!')
print('\n📋 Login Credentials:')
print('-' * 40)
print('ADMIN:')
print('  Email: admin@smartcare.com')
print('  Password: admin123')
print('  Role: Admin')
print()
print('DOCTORS:')
print('  doctor1@smartcare.com | doctor123')
print('  doctor2@smartcare.com | doctor456')
print('  drsindhya@smartcare.com | sindhya123')
print('  nurse@smartcare.com | nurse123')
conn.close()
