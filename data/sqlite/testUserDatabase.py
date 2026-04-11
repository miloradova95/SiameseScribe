import unittest
import uuid
from pathlib import Path

from userDatabase import UserDatabase


class UserDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dbPath = (
            Path(__file__).resolve().parent / f"testusers{uuid.uuid4().hex}.sqlite3"
        )
        self.database = UserDatabase(self.dbPath)

    def tearDown(self) -> None:
        del self.database
        if self.dbPath.exists():
            self.dbPath.unlink()

    def testCreateAndFetchUser(self) -> None:
        createdUser = self.database.createUser(
            username="alice",
            email="alice@example.com",
            password="super-secret",
        )

        fetchedUser = self.database.getUserByEmail("alice@example.com")

        self.assertIsNotNone(fetchedUser)
        self.assertEqual(createdUser.id, fetchedUser.id)
        self.assertNotEqual("super-secret", fetchedUser.password_hash)
        self.assertTrue(
            self.database.verifyPassword("super-secret", fetchedUser.password_hash)
        )

    def testDuplicateUserRaisesValueError(self) -> None:
        self.database.createUser(
            username="alice",
            email="alice@example.com",
            password="super-secret",
        )

        with self.assertRaises(ValueError):
            self.database.createUser(
                username="alice",
                email="alice-2@example.com",
                password="another-secret",
            )

    def testAuthenticateUserUpdatesLastLogin(self) -> None:
        self.database.createUser(
            username="alice",
            email="alice@example.com",
            password="super-secret",
        )

        authenticatedUser = self.database.authenticateUser("alice", "super-secret")

        self.assertIsNotNone(authenticatedUser)
        self.assertIsNotNone(authenticatedUser.last_login_at)

    def testGetAllUsersReturnsExistingUsers(self) -> None:
        self.database.createUser(
            username="alice",
            email="alice@example.com",
            password="super-secret",
        )
        self.database.createUser(
            username="bob",
            email="bob@example.com",
            password="another-secret",
        )

        allUsers = self.database.getAllUsers()

        self.assertEqual(2, len(allUsers))
        self.assertEqual("alice", allUsers[0].username)
        self.assertEqual("bob", allUsers[1].username)


if __name__ == "__main__":
    unittest.main()
