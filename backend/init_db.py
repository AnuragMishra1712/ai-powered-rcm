from backend.db import init_db

if __name__ == "__main__":
    print("🚀 Initializing Supabase database...")
    init_db()
    print("✅ Tables created successfully!")
