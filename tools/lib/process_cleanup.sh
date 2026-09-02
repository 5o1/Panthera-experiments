#!/usr/bin/env bash

# Shared process-group cleanup primitive for launch runners and shell tests.

stop_process_group() {
    local process_group_pid="${1:-}"
    local signal_name="${2:-TERM}"
    if [[ -z "$process_group_pid" ]] || ! kill -0 -- "-$process_group_pid" 2>/dev/null; then
        return 0
    fi
    kill -"$signal_name" -- "-$process_group_pid" 2>/dev/null || true
    for _cleanup_index in $(seq 1 30); do
        kill -0 -- "-$process_group_pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 -- "-$process_group_pid" 2>/dev/null; then
        kill -KILL -- "-$process_group_pid" 2>/dev/null || true
    fi
    wait "$process_group_pid" 2>/dev/null || true
}
