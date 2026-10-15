from probe_accessibility import run
from mindset.methods.routed_accessibility import RoutedAccessibilityMemory

if __name__ == '__main__':
    run(output_name='routed_accessibility_r01', memory_class=RoutedAccessibilityMemory,
        replay_reference=False)
