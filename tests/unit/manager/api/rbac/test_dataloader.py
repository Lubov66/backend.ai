"""Tests for RBAC role DataLoader utilities."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from ai.backend.manager.api.gql.data_loader.rbac.loader import (
    load_role_assignments_by_user_ids,
)
from ai.backend.manager.data.permission.role import AssignedUserData


class TestLoadRoleAssignmentsByUserIds:
    """Tests for load_role_assignments_by_user_ids function."""

    @staticmethod
    def create_mock_assignment(user_id: uuid.UUID, role_id: uuid.UUID | None = None) -> MagicMock:
        mock = MagicMock(spec=AssignedUserData)
        mock.user_id = user_id
        mock.role_id = role_id or uuid.uuid4()
        mock.id = uuid.uuid4()
        return mock

    @staticmethod
    def create_mock_processor(assignments: list[MagicMock]) -> MagicMock:
        mock_processor = MagicMock()
        mock_action_result = MagicMock()
        mock_action_result.result.items = assignments
        mock_processor.search_users_assigned_to_role.wait_for_complete = AsyncMock(
            return_value=mock_action_result
        )
        return mock_processor

    async def test_empty_ids_returns_empty_list(self) -> None:
        # Given
        mock_processor = MagicMock()

        # When
        result = await load_role_assignments_by_user_ids(mock_processor, [])

        # Then
        assert result == []
        mock_processor.search_users_assigned_to_role.wait_for_complete.assert_not_called()

    async def test_returns_assignments_grouped_by_user(self) -> None:
        # Given
        user1_id, user2_id = uuid.uuid4(), uuid.uuid4()
        assignment1 = self.create_mock_assignment(user1_id)
        assignment2 = self.create_mock_assignment(user1_id)
        assignment3 = self.create_mock_assignment(user2_id)
        mock_processor = self.create_mock_processor([assignment1, assignment2, assignment3])

        # When
        result = await load_role_assignments_by_user_ids(mock_processor, [user1_id, user2_id])

        # Then
        assert len(result) == 2
        assert result[0] == [assignment1, assignment2]
        assert result[1] == [assignment3]

    async def test_returns_empty_list_for_user_with_no_assignments(self) -> None:
        # Given
        user_with_roles = uuid.uuid4()
        user_without_roles = uuid.uuid4()
        assignment = self.create_mock_assignment(user_with_roles)
        mock_processor = self.create_mock_processor([assignment])

        # When
        result = await load_role_assignments_by_user_ids(
            mock_processor, [user_with_roles, user_without_roles]
        )

        # Then
        assert len(result) == 2
        assert result[0] == [assignment]
        assert result[1] == []

    async def test_preserves_input_order(self) -> None:
        # Given
        user1_id, user2_id, user3_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        a1 = self.create_mock_assignment(user3_id)
        a2 = self.create_mock_assignment(user1_id)
        mock_processor = self.create_mock_processor(
            [a1, a2]  # DB returns in different order
        )

        # When
        result = await load_role_assignments_by_user_ids(
            mock_processor, [user1_id, user2_id, user3_id]
        )

        # Then
        assert len(result) == 3
        assert result[0] == [a2]  # user1
        assert result[1] == []  # user2 (no assignments)
        assert result[2] == [a1]  # user3
