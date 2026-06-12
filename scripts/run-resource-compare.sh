#!/usr/bin/env bash
# Compare process-level resource usage (peak RSS, user/sys CPU) of the same
# aggregation query in a Quantum-enabled vs vanilla Spark JVM. Each mode
# runs in its own java process measured by /usr/bin/time -l.
#
# Usage: run-resource-compare.sh [rows] [groups] [iterations]
set -euo pipefail
cd "$(dirname "$0")/../java"

ROWS="${1:-10000000}"
NGROUPS="${2:-1000}"
ITERS="${3:-3}"
QUERY="${4:-agg}"

if [[ -z "${JAVA_HOME:-}" && -d /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ]]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi

NATIVE_DIR="${QUANTUM_NATIVE_DIR:-}"
if [[ -z "$NATIVE_DIR" ]]; then
  if [[ -e "$PWD/../build-release/src/libquantum_core.dylib" ]]; then
    NATIVE_DIR="$PWD/../build-release/src"
  else
    NATIVE_DIR="$PWD/../build-debug/src"
  fi
fi

# Full classpath including provided-scope Spark (java is invoked directly so
# Maven's own JVM doesn't pollute the measurement).
mvn -q compile dependency:build-classpath -DincludeScope=test \
    -Dmdep.outputFile=target/classpath.txt
CP="target/classes:$(cat target/classpath.txt)"

JVM_FLAGS=(
  -Djna.library.path="$NATIVE_DIR"
  -Dio.netty.tryReflectionSetAccessible=true
  --add-opens=java.base/java.lang=ALL-UNNAMED
  --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
  --add-opens=java.base/java.io=ALL-UNNAMED
  --add-opens=java.base/java.net=ALL-UNNAMED
  --add-opens=java.base/java.nio=ALL-UNNAMED
  --add-opens=java.base/java.util=ALL-UNNAMED
  --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
  --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
  --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
  --add-opens=java.base/sun.security.action=ALL-UNNAMED
  --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
)

DATA_DIR="$(mktemp -d)/t.parquet"
echo "rows=$ROWS groups=$NGROUPS iterations=$ITERS query=$QUERY"
"$JAVA_HOME/bin/java" "${JVM_FLAGS[@]}" -cp "$CP" \
  quantum.demo.QuantumResourceProbe prep "$DATA_DIR" "$ROWS" "$NGROUPS" > /dev/null 2>&1

run_mode() {
  local mode="$1"
  local tmpout; tmpout=$(mktemp)
  /usr/bin/time -l "$JAVA_HOME/bin/java" "${JVM_FLAGS[@]}" -cp "$CP" \
      quantum.demo.QuantumResourceProbe "$mode" "$DATA_DIR" "$ROWS" "$NGROUPS" "$ITERS" "$QUERY" \
      > "$tmpout" 2>&1 || true
  local result rss rss_mb cputimes
  result=$(grep "^RESULT" "$tmpout" || echo "RESULT mode=$mode FAILED")
  rss=$(awk '/maximum resident set size/{print $1; exit}' "$tmpout")
  rss_mb="?"
  [[ -n "$rss" ]] && rss_mb=$(( rss / 1048576 ))
  cputimes=$(awk '/real .*user .*sys/{$1=$1; print; exit}' "$tmpout")
  echo "$result peak_rss_mb=$rss_mb  $cputimes"
  rm -f "$tmpout"
}

echo "--- quantum ---"
run_mode quantum
echo "--- vanilla ---"
run_mode vanilla
rm -rf "$(dirname "$DATA_DIR")"
