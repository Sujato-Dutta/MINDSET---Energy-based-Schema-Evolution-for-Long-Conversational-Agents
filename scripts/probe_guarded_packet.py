"""Guarded variant using the immutable same-budget packet experiment harness.

The substitution is scoped to this standalone worker process. No source file,
baseline, queued job or original experiment is modified. Both wrapper and
shared harness hashes are recorded in the audit.
"""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import probe_packet_energy as harness
from mindset.methods.guarded_packet import GuardedPacketMindsetMemory
from mindset.acl.io import digest_file

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True); args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    writer = harness.write_json
    def audited_write(path, payload, **kwargs):
        if Path(path).name == 'audit.json':
            payload = {**payload, 'guarded_wrapper_sha256': digest_file(Path(__file__)),
                'shared_harness_sha256': digest_file(ROOT / 'scripts/probe_packet_energy.py'),
                'guard_source_sha256': digest_file(ROOT / 'src/mindset/methods/guarded_packet.py'),
                'implementation': 'guarded_packet_v2_development',
                'guard': 'retain top1, CE surrogate regret <= unchanged independent switch margin; no native-error guarantee'}
        return writer(path, payload, **kwargs)
    harness.PacketEnergyMindsetMemory = GuardedPacketMindsetMemory
    harness.write_json = audited_write
    harness.run(args.manifest, args.calibration, args.output)
