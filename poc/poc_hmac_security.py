import sqlite3
import hmac
import hashlib
import json
import datetime

# This secret key would normally be in an environment variable, NEVER in the DB
SECRET_KEY = b"super_secret_offline_key_for_fyp"

def generate_signature(transaction_data):
    # Combine the important fields to create the payload for the HMAC
    payload = f"{transaction_data['id']}|{transaction_data['amount']}|{transaction_data['date']}|{transaction_data['user_id']}"
    # Generate SHA-256 HMAC
    signature = hmac.new(SECRET_KEY, payload.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def setup_database(cursor):
    # Create a mock transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            amount REAL,
            date TEXT,
            user_id TEXT,
            signature TEXT
        )
    ''')

def insert_valid_transaction(conn, cursor):
    print("--- 1. Application Layer: Inserting Valid Transaction ---")
    tx_data = {
        'id': 101,
        'amount': 15000.0,
        'date': datetime.datetime.now().isoformat(),
        'user_id': "bookkeeper_ali"
    }
    
    sig = generate_signature(tx_data)
    
    cursor.execute('''
        INSERT INTO transactions (id, amount, date, user_id, signature)
        VALUES (?, ?, ?, ?, ?)
    ''', (tx_data['id'], tx_data['amount'], tx_data['date'], tx_data['user_id'], sig))
    conn.commit()
    print(f"Transaction inserted successfully.")
    print(f"Amount: {tx_data['amount']}")
    print(f"Generated Signature: {sig}")

def simulate_malicious_tamper(conn, cursor):
    print("\n--- 2. Malicious Actor: Bypassing App and Editing DB Directly ---")
    print("Bookkeeper Ali opens the database using raw SQL and changes 15,000 to 150,000")
    
    # Notice we don't update the signature, because the attacker doesn't know the SECRET_KEY
    cursor.execute('''
        UPDATE transactions 
        SET amount = 150000.0 
        WHERE id = 101
    ''')
    conn.commit()
    print("Database updated! The amount is now 150,000.")

def run_audit_check(cursor):
    print("\n--- 3. Audit System: Running Integrity Check ---")
    cursor.execute("SELECT id, amount, date, user_id, signature FROM transactions")
    rows = cursor.fetchall()
    
    tamper_detected = False
    for row in rows:
        tx_id, amount, date, user_id, stored_sig = row
        tx_data = {'id': tx_id, 'amount': amount, 'date': date, 'user_id': user_id}
        
        # Recalculate signature using the secret key
        expected_sig = generate_signature(tx_data)
        
        if expected_sig != stored_sig:
            print("================================================================")
            print(f"*** TAMPER DETECTED: Signature Mismatch on Transaction {tx_id}! ***")
            print(f"Stored Signature : {stored_sig}")
            print(f"Expected Signatre: {expected_sig}")
            print(f"Current Amount   : {amount}")
            print("================================================================")
            tamper_detected = True
        else:
            print(f"Row {tx_id} is secure and verified.")
            
    if not tamper_detected:
        print("Audit complete. All records are secure.")

if __name__ == "__main__":
    print("==================================================")
    print("POC 3: Row-Level HMAC Tamper Detection Demo")
    print("==================================================")
    
    # Use an in-memory database for the quick POC
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    setup_database(cursor)
    
    # The normal flow
    insert_valid_transaction(conn, cursor)
    
    # The attack
    simulate_malicious_tamper(conn, cursor)
    
    # The detection
    run_audit_check(cursor)
    
    conn.close()
