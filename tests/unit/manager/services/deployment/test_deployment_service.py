"""
Mock-based unit tests for DeploymentService.

Tests verify service layer business logic using mocked repositories.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.data.endpoint.types import EndpointLifecycle
from ai.backend.common.data.model_deployment.types import DeploymentStrategy
from ai.backend.common.types import ClusterMode, ResourceSlot, RuntimeVariant
from ai.backend.manager.data.deployment.creator import ModelRevisionCreator, VFolderMountsCreator
from ai.backend.manager.data.deployment.types import (
    ClusterConfigData,
    DeploymentInfo,
    DeploymentMetadata,
    DeploymentNetworkSpec,
    DeploymentPolicyData,
    DeploymentPolicySearchResult,
    DeploymentState,
    ExecutionSpec,
    ModelMountConfigData,
    ModelRevisionData,
    ModelRuntimeConfigData,
    ModelServiceDefinition,
    ReplicaSpec,
    ResourceConfigData,
    ResourceSpec,
)
from ai.backend.manager.models.deployment_policy import RollingUpdateSpec
from ai.backend.manager.repositories.base import BatchQuerier, OffsetPagination
from ai.backend.manager.repositories.deployment import DeploymentRepository
from ai.backend.manager.services.deployment.actions.deployment_policy import (
    SearchDeploymentPoliciesAction,
)
from ai.backend.manager.services.deployment.actions.model_revision.add_model_revision import (
    AddModelRevisionAction,
)
from ai.backend.manager.services.deployment.actions.model_revision.create_model_revision import (
    CreateModelRevisionAction,
)
from ai.backend.manager.services.deployment.processors import DeploymentProcessors
from ai.backend.manager.services.deployment.service import DeploymentService
from ai.backend.manager.sokovan.deployment import DeploymentController
from ai.backend.manager.sokovan.deployment.revision_generator.registry import (
    RevisionGeneratorRegistry,
)


class DeploymentServiceBaseFixtures:
    """Base class containing shared fixtures for deployment service tests."""

    @pytest.fixture
    def mock_deployment_repository(self) -> MagicMock:
        """Mock DeploymentRepository for testing."""
        return MagicMock(spec=DeploymentRepository)

    @pytest.fixture
    def mock_deployment_controller(self) -> MagicMock:
        """Mock DeploymentController for testing."""
        return MagicMock(spec=DeploymentController)

    @pytest.fixture
    def mock_revision_generator_registry(self) -> MagicMock:
        """Mock RevisionGeneratorRegistry for testing."""
        return MagicMock(spec=RevisionGeneratorRegistry)

    @pytest.fixture
    def deployment_service(
        self,
        mock_deployment_controller: MagicMock,
        mock_deployment_repository: MagicMock,
        mock_revision_generator_registry: MagicMock,
    ) -> DeploymentService:
        """Create DeploymentService with mock dependencies."""
        return DeploymentService(
            deployment_controller=mock_deployment_controller,
            deployment_repository=mock_deployment_repository,
            revision_generator_registry=mock_revision_generator_registry,
        )

    @pytest.fixture
    def processors(self, deployment_service: DeploymentService) -> DeploymentProcessors:
        """Create DeploymentProcessors with mock DeploymentService."""
        return DeploymentProcessors(deployment_service, [])

    @pytest.fixture
    def deployment_policy_data(self) -> DeploymentPolicyData:
        """Sample deployment policy data for testing."""
        return DeploymentPolicyData(
            id=uuid.uuid4(),
            endpoint=uuid.uuid4(),
            strategy=DeploymentStrategy.ROLLING,
            strategy_spec=RollingUpdateSpec(max_surge=1, max_unavailable=0),
            rollback_on_failure=False,
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            updated_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        )


class TestSearchDeploymentPolicies(DeploymentServiceBaseFixtures):
    """Tests for DeploymentService.search_deployment_policies"""

    async def test_search_deployment_policies_success(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_policy_data: DeploymentPolicyData,
    ) -> None:
        """Search deployment policies should return matching results."""
        mock_deployment_repository.search_deployment_policies = AsyncMock(
            return_value=DeploymentPolicySearchResult(
                items=[deployment_policy_data],
                total_count=1,
                has_next_page=False,
                has_previous_page=False,
            )
        )

        querier = BatchQuerier(
            pagination=OffsetPagination(limit=10, offset=0),
            conditions=[],
            orders=[],
        )
        action = SearchDeploymentPoliciesAction(querier=querier)

        result = await processors.search_deployment_policies.wait_for_complete(action)

        assert result.data == [deployment_policy_data]
        assert result.total_count == 1
        assert result.has_next_page is False
        assert result.has_previous_page is False
        mock_deployment_repository.search_deployment_policies.assert_called_once_with(querier)

    async def test_search_deployment_policies_empty_result(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
    ) -> None:
        """Search deployment policies should return empty list when no results found."""
        mock_deployment_repository.search_deployment_policies = AsyncMock(
            return_value=DeploymentPolicySearchResult(
                items=[],
                total_count=0,
                has_next_page=False,
                has_previous_page=False,
            )
        )

        querier = BatchQuerier(
            pagination=OffsetPagination(limit=10, offset=0),
            conditions=[],
            orders=[],
        )
        action = SearchDeploymentPoliciesAction(querier=querier)

        result = await processors.search_deployment_policies.wait_for_complete(action)

        assert result.data == []
        assert result.total_count == 0

    async def test_search_deployment_policies_with_pagination(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_policy_data: DeploymentPolicyData,
    ) -> None:
        """Search deployment policies should handle pagination correctly."""
        mock_deployment_repository.search_deployment_policies = AsyncMock(
            return_value=DeploymentPolicySearchResult(
                items=[deployment_policy_data],
                total_count=25,
                has_next_page=True,
                has_previous_page=True,
            )
        )

        querier = BatchQuerier(
            pagination=OffsetPagination(limit=10, offset=10),
            conditions=[],
            orders=[],
        )
        action = SearchDeploymentPoliciesAction(querier=querier)

        result = await processors.search_deployment_policies.wait_for_complete(action)

        assert result.total_count == 25
        assert result.has_next_page is True
        assert result.has_previous_page is True


class ModelRevisionFixtures(DeploymentServiceBaseFixtures):
    """Fixtures for model revision tests."""

    @pytest.fixture(autouse=True)
    def _setup_revision_generator(self, mock_revision_generator_registry: MagicMock) -> None:
        """Set up mock revision generator to return no service definition by default."""
        mock_generator = MagicMock()
        mock_generator.load_service_definition = AsyncMock(return_value=None)
        mock_revision_generator_registry.get.return_value = mock_generator

    @pytest.fixture(autouse=True)
    def _setup_default_repository_mocks(
        self,
        mock_deployment_repository: MagicMock,
        endpoint_info: DeploymentInfo,
        revision_data: ModelRevisionData,
    ) -> None:
        """Set up default mock responses for repository methods used in _build_revision."""
        mock_deployment_repository.get_endpoint_info = AsyncMock(return_value=endpoint_info)
        mock_deployment_repository.get_latest_revision_number = AsyncMock(return_value=None)
        mock_deployment_repository.create_revision = AsyncMock(return_value=revision_data)

    @pytest.fixture
    def deployment_id(self) -> uuid.UUID:
        return uuid.uuid4()

    @pytest.fixture
    def image_id(self) -> uuid.UUID:
        return uuid.uuid4()

    @pytest.fixture
    def model_vfolder_id(self) -> uuid.UUID:
        return uuid.uuid4()

    @pytest.fixture
    def endpoint_info(self, deployment_id: uuid.UUID) -> DeploymentInfo:
        return DeploymentInfo(
            id=deployment_id,
            metadata=DeploymentMetadata(
                name="test-deployment",
                domain="default",
                project=uuid.uuid4(),
                resource_group="default",
                created_user=uuid.uuid4(),
                session_owner=uuid.uuid4(),
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
                revision_history_limit=10,
            ),
            state=DeploymentState(
                lifecycle=EndpointLifecycle.READY,
                retry_count=0,
            ),
            replica_spec=ReplicaSpec(replica_count=1),
            network=DeploymentNetworkSpec(open_to_public=False),
            model_revisions=[],
        )

    @pytest.fixture
    def revision_creator(
        self, image_id: uuid.UUID, model_vfolder_id: uuid.UUID
    ) -> ModelRevisionCreator:
        return ModelRevisionCreator(
            image_id=image_id,
            resource_group_name="default",
            resource_spec=ResourceSpec(
                cluster_mode=ClusterMode.SINGLE_NODE,
                cluster_size=1,
                resource_slots={"cpu": "4", "mem": "8g"},
                resource_opts={"shmem": "1g"},
            ),
            mounts=VFolderMountsCreator(
                model_vfolder_id=model_vfolder_id,
                model_definition_path="model-definition.yaml",
                model_mount_destination="/models",
            ),
            execution=ExecutionSpec(
                startup_command="python serve.py",
                bootstrap_script="pip install -r requirements.txt",
                environ={"CUDA_VISIBLE_DEVICES": "0"},
                runtime_variant=RuntimeVariant.VLLM,
                callback_url=None,
            ),
        )

    @pytest.fixture
    def revision_data(self, image_id: uuid.UUID, model_vfolder_id: uuid.UUID) -> ModelRevisionData:
        return ModelRevisionData(
            id=uuid.uuid4(),
            name="rev-1",
            cluster_config=ClusterConfigData(
                mode=ClusterMode.SINGLE_NODE,
                size=1,
            ),
            resource_config=ResourceConfigData(
                resource_group_name="default",
                resource_slot=ResourceSlot({"cpu": "4", "mem": "8g"}),
            ),
            model_runtime_config=ModelRuntimeConfigData(
                runtime_variant=RuntimeVariant.VLLM,
            ),
            model_mount_config=ModelMountConfigData(
                vfolder_id=model_vfolder_id,
                mount_destination="/models",
                definition_path="model-definition.yaml",
            ),
            image_id=image_id,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

    @pytest.fixture
    def revision_creator_with_none_environ(
        self, image_id: uuid.UUID, model_vfolder_id: uuid.UUID
    ) -> ModelRevisionCreator:
        """Creator with None environ and resource_opts for edge case testing."""
        return ModelRevisionCreator(
            image_id=image_id,
            resource_group_name="default",
            resource_spec=ResourceSpec(
                cluster_mode=ClusterMode.SINGLE_NODE,
                cluster_size=1,
                resource_slots={"cpu": "2"},
                resource_opts=None,
            ),
            mounts=VFolderMountsCreator(
                model_vfolder_id=model_vfolder_id,
            ),
            execution=ExecutionSpec(
                runtime_variant=RuntimeVariant.VLLM,
                environ=None,
            ),
        )


class TestAddModelRevision(ModelRevisionFixtures):
    """Tests for DeploymentService.add_model_revision"""

    async def test_add_model_revision_first_revision(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        endpoint_info: DeploymentInfo,
        revision_creator: ModelRevisionCreator,
        revision_data: ModelRevisionData,
    ) -> None:
        """Adding the first revision should use revision_number=1."""
        action = AddModelRevisionAction(deployment_id=deployment_id, adder=revision_creator)
        result = await processors.add_model_revision.wait_for_complete(action)

        assert result.revision == revision_data
        mock_deployment_repository.get_endpoint_info.assert_called_once_with(deployment_id)
        mock_deployment_repository.get_latest_revision_number.assert_called_once_with(deployment_id)
        mock_deployment_repository.create_revision.assert_called_once()

        call_args = mock_deployment_repository.create_revision.call_args
        assert call_args[0][0] == deployment_id
        creator_arg = call_args[0][1]
        spec = creator_arg.spec
        assert spec.revision_number == 1
        assert spec.image_id == revision_creator.image_id
        assert spec.resource_group == endpoint_info.metadata.resource_group
        assert spec.model_id == revision_creator.mounts.model_vfolder_id
        assert spec.runtime_variant == RuntimeVariant.VLLM

    async def test_add_model_revision_increments_revision_number(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator: ModelRevisionCreator,
        revision_data: ModelRevisionData,
    ) -> None:
        """Adding a revision when revisions exist should increment the number."""
        mock_deployment_repository.get_latest_revision_number = AsyncMock(return_value=3)

        action = AddModelRevisionAction(deployment_id=deployment_id, adder=revision_creator)
        result = await processors.add_model_revision.wait_for_complete(action)

        assert result.revision == revision_data
        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        assert creator_arg.spec.revision_number == 4

    async def test_add_model_revision_maps_resource_fields(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator: ModelRevisionCreator,
    ) -> None:
        """All fields from ModelRevisionCreator should be mapped to the spec correctly."""
        action = AddModelRevisionAction(deployment_id=deployment_id, adder=revision_creator)
        await processors.add_model_revision.wait_for_complete(action)

        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        spec = creator_arg.spec
        assert spec.resource_slots == ResourceSlot(revision_creator.resource_spec.resource_slots)
        assert spec.resource_opts == revision_creator.resource_spec.resource_opts
        assert spec.cluster_mode == revision_creator.resource_spec.cluster_mode
        assert spec.cluster_size == revision_creator.resource_spec.cluster_size
        assert spec.model_mount_destination == revision_creator.mounts.model_mount_destination
        assert spec.model_definition_path == revision_creator.mounts.model_definition_path
        assert spec.model_definition is None
        assert spec.startup_command == revision_creator.execution.startup_command
        assert spec.bootstrap_script == revision_creator.execution.bootstrap_script
        assert spec.environ == revision_creator.execution.environ
        assert spec.callback_url is None
        assert spec.extra_mounts == []

    async def test_add_model_revision_with_none_environ(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator_with_none_environ: ModelRevisionCreator,
    ) -> None:
        """None environ and resource_opts should be converted to empty dict."""
        action = AddModelRevisionAction(
            deployment_id=deployment_id, adder=revision_creator_with_none_environ
        )
        await processors.add_model_revision.wait_for_complete(action)

        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        spec = creator_arg.spec
        assert spec.environ == {}
        assert spec.resource_opts == {}


class TestCreateModelRevision(ModelRevisionFixtures):
    """Tests for DeploymentService.create_model_revision

    create_model_revision creates orphan revisions (not attached to any deployment).
    """

    @pytest.fixture
    def orphan_revision_creator(
        self, image_id: uuid.UUID, model_vfolder_id: uuid.UUID
    ) -> ModelRevisionCreator:
        """Creator for orphan revision (no deployment)."""
        return ModelRevisionCreator(
            image_id=image_id,
            resource_group_name="my-rg",
            resource_spec=ResourceSpec(
                cluster_mode=ClusterMode.SINGLE_NODE,
                cluster_size=1,
                resource_slots={"cpu": "4", "mem": "8g"},
                resource_opts={"shmem": "1g"},
            ),
            mounts=VFolderMountsCreator(
                model_vfolder_id=model_vfolder_id,
                model_definition_path="model-definition.yaml",
                model_mount_destination="/models",
            ),
            execution=ExecutionSpec(
                startup_command="python serve.py",
                bootstrap_script="pip install -r requirements.txt",
                environ={"CUDA_VISIBLE_DEVICES": "0"},
                runtime_variant=RuntimeVariant.VLLM,
                callback_url=None,
            ),
        )

    @pytest.fixture
    def orphan_revision_creator_with_none_environ(
        self, image_id: uuid.UUID, model_vfolder_id: uuid.UUID
    ) -> ModelRevisionCreator:
        """Orphan creator with None environ and resource_opts for edge case testing."""
        return ModelRevisionCreator(
            image_id=image_id,
            resource_group_name="my-rg",
            resource_spec=ResourceSpec(
                cluster_mode=ClusterMode.SINGLE_NODE,
                cluster_size=1,
                resource_slots={"cpu": "2"},
                resource_opts=None,
            ),
            mounts=VFolderMountsCreator(
                model_vfolder_id=model_vfolder_id,
            ),
            execution=ExecutionSpec(
                runtime_variant=RuntimeVariant.VLLM,
                environ=None,
            ),
        )

    async def test_create_model_revision_creates_orphan(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        orphan_revision_creator: ModelRevisionCreator,
        revision_data: ModelRevisionData,
    ) -> None:
        """Creating a revision should create an orphan with endpoint_id=None."""
        action = CreateModelRevisionAction(creator=orphan_revision_creator)
        result = await processors.create_model_revision.wait_for_complete(action)

        assert result.revision == revision_data
        mock_deployment_repository.get_endpoint_info.assert_not_called()
        mock_deployment_repository.get_latest_revision_number.assert_not_called()
        mock_deployment_repository.create_revision.assert_called_once()

        call_args = mock_deployment_repository.create_revision.call_args
        assert call_args[0][0] is None
        creator_arg = call_args[0][1]
        spec = creator_arg.spec
        assert spec.revision_number == 0
        assert spec.image_id == orphan_revision_creator.image_id

    async def test_create_model_revision_uses_creator_resource_group(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        orphan_revision_creator: ModelRevisionCreator,
    ) -> None:
        """resource_group should come from the creator's resource_group_name."""
        action = CreateModelRevisionAction(creator=orphan_revision_creator)
        await processors.create_model_revision.wait_for_complete(action)

        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        assert creator_arg.spec.resource_group == "my-rg"

    async def test_create_model_revision_maps_all_fields(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        orphan_revision_creator: ModelRevisionCreator,
    ) -> None:
        """All fields from ModelRevisionCreator should be mapped to the spec correctly."""
        action = CreateModelRevisionAction(creator=orphan_revision_creator)
        await processors.create_model_revision.wait_for_complete(action)

        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        spec = creator_arg.spec
        assert spec.resource_slots == ResourceSlot(
            orphan_revision_creator.resource_spec.resource_slots
        )
        assert spec.resource_opts == orphan_revision_creator.resource_spec.resource_opts
        assert spec.cluster_mode == orphan_revision_creator.resource_spec.cluster_mode
        assert spec.cluster_size == orphan_revision_creator.resource_spec.cluster_size
        assert spec.model_id == orphan_revision_creator.mounts.model_vfolder_id
        assert (
            spec.model_mount_destination == orphan_revision_creator.mounts.model_mount_destination
        )
        assert spec.model_definition_path == orphan_revision_creator.mounts.model_definition_path
        assert spec.model_definition is None
        assert spec.startup_command == orphan_revision_creator.execution.startup_command
        assert spec.bootstrap_script == orphan_revision_creator.execution.bootstrap_script
        assert spec.environ == orphan_revision_creator.execution.environ
        assert spec.callback_url is None
        assert spec.runtime_variant == RuntimeVariant.VLLM
        assert spec.extra_mounts == []

    async def test_create_model_revision_with_none_environ(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        orphan_revision_creator_with_none_environ: ModelRevisionCreator,
    ) -> None:
        """None environ and resource_opts should be converted to empty dict."""
        action = CreateModelRevisionAction(creator=orphan_revision_creator_with_none_environ)
        await processors.create_model_revision.wait_for_complete(action)

        creator_arg = mock_deployment_repository.create_revision.call_args[0][1]
        spec = creator_arg.spec
        assert spec.environ == {}
        assert spec.resource_opts == {}


class TestServiceDefinitionMerge(ModelRevisionFixtures):
    """Tests for service definition merging in revision creation."""

    @pytest.fixture
    def setup_mock_service_definition(
        self, mock_revision_generator_registry: MagicMock
    ) -> Callable[[ModelServiceDefinition], None]:
        """Factory fixture to inject a service definition into the mock generator registry."""

        def _setup(service_def: ModelServiceDefinition) -> None:
            mock_generator = MagicMock()
            mock_generator.load_service_definition = AsyncMock(return_value=service_def)
            mock_revision_generator_registry.get.return_value = mock_generator

        return _setup

    async def test_merge_environ_from_service_definition(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator: ModelRevisionCreator,
        setup_mock_service_definition: Callable[[ModelServiceDefinition], None],
    ) -> None:
        """Service definition environ should be merged with creator environ as base."""
        setup_mock_service_definition(
            ModelServiceDefinition(
                environ={"SERVICE_VAR": "from_def", "CUDA_VISIBLE_DEVICES": "1"},
            )
        )

        action = AddModelRevisionAction(deployment_id=deployment_id, adder=revision_creator)
        await processors.add_model_revision.wait_for_complete(action)

        spec = mock_deployment_repository.create_revision.call_args[0][1].spec
        # Creator value overrides service definition for overlapping keys
        assert spec.environ["CUDA_VISIBLE_DEVICES"] == "0"
        # Service definition provides new keys
        assert spec.environ["SERVICE_VAR"] == "from_def"

    async def test_merge_resource_slots_from_service_definition(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        revision_creator: ModelRevisionCreator,
        setup_mock_service_definition: Callable[[ModelServiceDefinition], None],
    ) -> None:
        """Service definition resource_slots should be merged with creator slots as base."""
        setup_mock_service_definition(
            ModelServiceDefinition(
                resource_slots={"cpu": "2", "mem": "4g", "cuda.shares": "1.0"},
            )
        )

        action = CreateModelRevisionAction(creator=revision_creator)
        await processors.create_model_revision.wait_for_complete(action)

        spec = mock_deployment_repository.create_revision.call_args[0][1].spec
        expected = ResourceSlot({"cpu": "4", "mem": "8g", "cuda.shares": "1.0"})
        assert spec.resource_slots == expected

    async def test_no_service_definition_uses_creator_values_as_is(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator: ModelRevisionCreator,
    ) -> None:
        """When no service definition exists, creator values are used unchanged."""
        action = AddModelRevisionAction(deployment_id=deployment_id, adder=revision_creator)
        await processors.add_model_revision.wait_for_complete(action)

        spec = mock_deployment_repository.create_revision.call_args[0][1].spec
        assert spec.environ == revision_creator.execution.environ
        assert spec.resource_slots == ResourceSlot(revision_creator.resource_spec.resource_slots)

    async def test_service_definition_with_empty_fields_no_effect(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        revision_creator: ModelRevisionCreator,
        setup_mock_service_definition: Callable[[ModelServiceDefinition], None],
    ) -> None:
        """Service definition with None environ/resource_slots should not affect creator."""
        setup_mock_service_definition(ModelServiceDefinition(environ=None, resource_slots=None))

        action = CreateModelRevisionAction(creator=revision_creator)
        await processors.create_model_revision.wait_for_complete(action)

        spec = mock_deployment_repository.create_revision.call_args[0][1].spec
        assert spec.environ == revision_creator.execution.environ
        assert spec.resource_slots == ResourceSlot(revision_creator.resource_spec.resource_slots)

    async def test_service_definition_environ_with_none_creator_environ(
        self,
        processors: DeploymentProcessors,
        mock_deployment_repository: MagicMock,
        deployment_id: uuid.UUID,
        revision_creator_with_none_environ: ModelRevisionCreator,
        setup_mock_service_definition: Callable[[ModelServiceDefinition], None],
    ) -> None:
        """Service definition environ should apply when creator environ is None."""
        setup_mock_service_definition(
            ModelServiceDefinition(
                environ={"SERVICE_VAR": "value"},
            )
        )

        action = AddModelRevisionAction(
            deployment_id=deployment_id, adder=revision_creator_with_none_environ
        )
        await processors.add_model_revision.wait_for_complete(action)

        spec = mock_deployment_repository.create_revision.call_args[0][1].spec
        assert spec.environ == {"SERVICE_VAR": "value"}
