"""El arranque no puede morir porque un parche opcional intente usar la red."""
import socket
import sys

import studio.compat_patches as cp
import studio.memory_safety_hooks as hooks


def _boom():
    socket.create_connection(("127.0.0.1", 1), timeout=0.2)  # ConnectionRefused, como en Colab


def test_peft_roto_no_tumba_el_arranque(monkeypatch):
    monkeypatch.setattr(cp, "_patch_peft_integrations", _boom)
    cp.apply_runtime_compatibility_patches()  # no debe lanzar


def test_hooks_rotos_no_tumban_el_arranque(monkeypatch):
    monkeypatch.setattr(hooks, "install_memory_safety_hooks", _boom)
    cp.apply_runtime_compatibility_patches()  # no debe lanzar


def test_peft_no_se_importa_si_no_estaba_cargado(monkeypatch):
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    cp._patch_peft_integrations()
    assert "peft" not in sys.modules, "importar peft al arrancar provocaba el fallo de red"


def test_launcher_detects_a_dead_child_immediately(monkeypatch):
    """Antes esperaba 35 s contra un proceso muerto y el error real quedaba enterrado."""
    import launch_colab

    class DeadProc:
        returncode = 1

        def poll(self):
            return 1

    import time

    started = time.time()
    assert launch_colab.wait_for_local_server(7999, timeout=30.0, proc=DeadProc()) is None
    assert time.time() - started < 2.0, "debe abortar en cuanto el hijo muere"
