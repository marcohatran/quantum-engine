#!/usr/bin/env bash
# Run the Quantum Engine demo job (quantum.demo.QuantumDemo): a Spark SQL
# aggregation executing on the native engine, printing results + the
# native query trace + engine counters.
#
# Prerequisites:
#   - native lib built:  cmake --preset=debug && cmake --build --preset=debug
#   - JDK 17+ on JAVA_HOME (or Homebrew openjdk@21), Maven
set -euo pipefail
cd "$(dirname "$0")/../java"

if [[ -z "${JAVA_HOME:-}" && -d /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ]]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi

NATIVE_LIB_DIR="$PWD/../build-debug/src"
if [[ ! -e "$NATIVE_LIB_DIR/libquantum_core.dylib" && ! -e "$NATIVE_LIB_DIR/libquantum_core.so" ]]; then
  echo "native library not found in $NATIVE_LIB_DIR — build it first:" >&2
  echo "  cmake --preset=debug && cmake --build --preset=debug" >&2
  exit 1
fi

# Spark on JDK 17+ needs the module opens; Arrow needs the netty flag.
export MAVEN_OPTS="-Djna.library.path=$NATIVE_LIB_DIR \
  -Dio.netty.tryReflectionSetAccessible=true \
  --add-opens=java.base/java.lang=ALL-UNNAMED \
  --add-opens=java.base/java.lang.invoke=ALL-UNNAMED \
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
  --add-opens=java.base/java.io=ALL-UNNAMED \
  --add-opens=java.base/java.net=ALL-UNNAMED \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  --add-opens=java.base/java.util=ALL-UNNAMED \
  --add-opens=java.base/java.util.concurrent=ALL-UNNAMED \
  --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED \
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens=java.base/sun.nio.cs=ALL-UNNAMED \
  --add-opens=java.base/sun.security.action=ALL-UNNAMED \
  --add-opens=java.base/sun.util.calendar=ALL-UNNAMED"

exec mvn -q compile exec:java \
  -Dexec.classpathScope=test \
  -Dexec.cleanupDaemonThreads=false \
  -Dexec.mainClass=quantum.demo.QuantumDemo
