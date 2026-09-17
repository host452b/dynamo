# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from kubernetes_asyncio.client import exceptions

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_logging_config_reads_existing_v1beta1_env(tmp_path) -> None:
    """Recognize JSONL logging already declared in a v1beta1 manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": {
            "components": [],
            "env": [{"name": "DYN_LOGGING_JSONL", "value": "1"}],
        },
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.get_logging_config()["jsonl_enabled"] is True


async def test_in_flight_restart_preserves_bounded_previous_log(tmp_path) -> None:
    """Keep a bounded previous-instance log before Kubernetes rotates again."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd"),
        namespace="default",
    )
    terminated = SimpleNamespace(reason="Error", exit_code=1)
    container_status = SimpleNamespace(
        name="main",
        restart_count=1,
        last_state=SimpleNamespace(terminated=terminated),
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="worker-0"),
        status=SimpleNamespace(container_statuses=[container_status]),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(return_value=SimpleNamespace(items=[pod])),
        read_namespaced_pod_log=AsyncMock(
            return_value="first line\nsecond line\nthird line\n"
        ),
    )

    warnings = await deployment._dump_in_flight_restart_logs(prev_log_tail_lines=2)

    assert len(warnings) == 1
    assert "first line" not in warnings[0]
    assert "second line" in warnings[0]
    assert "third line" in warnings[0]
    preserved = tmp_path / "restarts" / "worker-0.main.restart-1.previous.log"
    assert preserved.read_text() == "first line\nsecond line\nthird line\n"
    deployment._core_api.read_namespaced_pod_log.assert_awaited_once_with(
        name="worker-0",
        namespace="default",
        container="main",
        previous=True,
        tail_lines=50000,
    )


async def test_cleanup_preserves_checkpoint_pod_logs_before_deletion(
    tmp_path, monkeypatch
) -> None:
    """Capture a source pod without DGD labels before cascading deletion."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd", services=[]),
        namespace="default",
    )
    deployment._custom_api = SimpleNamespace(
        list_namespaced_custom_object=AsyncMock(
            return_value={
                "items": [{"metadata": {"name": "checkpoint-test", "uid": "job-uid"}}]
            }
        ),
    )
    pod = Mock()
    pod.name = "checkpoint-test-source"
    pod.raw = {
        "metadata": {
            "name": pod.name,
            "labels": {
                "nvidia.com/snapshot-job": "checkpoint-test",
                "nvidia.com/snapshot-job-uid": "job-uid",
            },
        },
        "spec": {"containers": [{"name": "main"}]},
    }
    pod.to_yaml.return_value = yaml.safe_dump(pod.raw)
    pod.logs.side_effect = lambda **kwargs: [
        "previous instance" if kwargs.get("previous") else "source startup error"
    ]

    def get_pods(kind, *, namespace, label_selector):
        assert kind == "pods"
        assert namespace == "default"
        if label_selector == (
            "nvidia.com/snapshot-job=checkpoint-test,"
            "nvidia.com/snapshot-job-uid=job-uid"
        ):
            return [pod]
        return []

    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    metrics = Mock(side_effect=AssertionError("source pods must not scrape metrics"))
    monkeypatch.setattr(deployment, "_get_pod_metrics", metrics)
    directory = tmp_path / "CheckpointSource"

    async def delete_deployment():
        assert (
            directory / f"{pod.name}.main.log"
        ).read_text() == "source startup error"
        assert (
            directory / f"{pod.name}.main.previous.log"
        ).read_text() == "previous instance"
        assert yaml.safe_load((directory / f"{pod.name}.yaml").read_text()) == pod.raw

    delete = AsyncMock(side_effect=delete_deployment)
    monkeypatch.setattr(deployment, "_delete_deployment", delete)
    await deployment._cleanup()
    deployment._custom_api.list_namespaced_custom_object.assert_awaited_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace="default",
        plural="snapshotjobs",
        label_selector="nvidia.com/dynamo-graph-deployment-name=test-dgd",
        _request_timeout=30,
    )
    delete.assert_awaited_once()
    metrics.assert_not_called()


@pytest.mark.parametrize("status", [404, 403, 500])
async def test_checkpoint_log_api_failure_does_not_prevent_cleanup(
    tmp_path, monkeypatch, status
) -> None:
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd", services=[]),
        namespace="default",
    )
    deployment._custom_api = SimpleNamespace(
        list_namespaced_custom_object=AsyncMock(
            side_effect=exceptions.ApiException(status=status)
        ),
    )
    delete = AsyncMock()
    monkeypatch.setattr(deployment, "_delete_deployment", delete)
    await deployment._cleanup()
    delete.assert_awaited_once()
