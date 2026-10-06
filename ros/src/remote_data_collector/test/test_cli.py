"""Tests for the `rdc` command line client, run in-process against a real collector node."""

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('remote_data_collector_interfaces')

from remote_data_collector import cli  # noqa: E402


def rdc(h, capsys, *argv):
    """Run `rdc ARGV` with the harness' client node; return (exit code, stdout, stderr)."""
    code = cli.run(['--timeout', '5', '--stop-timeout', '30'] + list(argv), h.client_node)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(['start'])  # missing remote
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(['frobnicate'])
    assert exc.value.code == 2


def test_cli_full_cycle(harness_factory, capsys, tmp_path):
    h = harness_factory()
    remote = '/' + h.name

    code, out, _ = rdc(h, capsys, 'status', remote)
    assert code == 0 and f'{remote} on ' in out and 'active:   0 / 1' in out

    code, out, _ = rdc(h, capsys, 'start', remote, '-n', 'clitest', '-p', 'tf', '-t', '/x')
    assert code == 0 and 'recording clitest_' in out
    bag = out.split('recording ')[1].split()[0]

    code, out, _ = rdc(h, capsys, 'status', remote)
    assert bag in out and '[RECORDING]' in out

    code, _, err = rdc(h, capsys, 'start', remote)
    assert code == 1 and 'busy' in err

    code, out, _ = rdc(h, capsys, 'list', remote)
    assert bag in out and 'RECORDING' in out

    dest = tmp_path / 'dl'
    code, out, _ = rdc(h, capsys, 'stop', remote, '--download', str(dest))
    assert code == 0, out
    assert 'CANCELED' in out and f'downloaded -> {dest / bag}' in out
    assert (dest / bag / 'metadata.yaml').is_file()

    code, out, _ = rdc(h, capsys, 'stop', remote)
    assert code == 0 and 'nothing was recording' in out

    dest2 = tmp_path / 'dl2'
    code, out, _ = rdc(h, capsys, 'download', remote, bag, '--dest', str(dest2), '--delete')
    assert code == 0 and (dest2 / bag).is_dir() and 'deleted on collector' in out
    code, out, _ = rdc(h, capsys, 'list', remote)
    assert '(no bags)' in out

    code, _, err = rdc(h, capsys, 'delete', remote, bag)
    assert code == 1 and 'no bag named' in err


def test_cli_timed_wait_download_delete(harness_factory, capsys, tmp_path):
    h = harness_factory()
    remote = '/' + h.name
    code, out, _ = rdc(h, capsys, 'start', remote, '-d', '1', '--wait', '--download',
                       str(tmp_path), '--delete')
    assert code == 0, out
    assert 'SUCCEEDED' in out and 'MISSING (1): /demo/missing_b' in out
    assert 'deleted on collector' in out
    assert h.client.list_bags() == []


def test_cli_unreachable_and_discover(harness_factory, capsys):
    h = harness_factory()
    code, _, err = rdc(h, capsys, 'status', '/no_such_collector')
    assert code == 3 and 'not reachable' in err
    code, out, _ = rdc(h, capsys, 'discover', '--wait', '0.5')
    assert code == 0 and f'/{h.name}' in out.split()
