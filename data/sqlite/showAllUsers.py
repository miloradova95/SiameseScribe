from userDatabase import createUserDatabase

def main() -> None:
    database = createUserDatabase()
    users = database.getAllUsers()

    if not users:
        print("No users found.")
        return

    print("Existing users:")
    for user in users:
        print(
            f"- id={user.id} username={user.username} email={user.email} "
            f"createdAt={user.created_at} lastLoginAt={user.last_login_at}"
        )


if __name__ == "__main__":
    main()
