#pragma once

#include <string>
#include <string_view>

#include "plan/Plan.hpp"

// Forward-declare the protobuf message types so callers don't pull in the
// generated .pb.h headers transitively.
namespace quantum::plan::proto {
class PlanNode;
class ScalarExpr;
}

namespace quantum::plan {

// Wire format: protobuf-encoded PlanNode messages. Compatible with the JVM
// bridge once it speaks the same .proto schema.

// Binary serialization round-trip.
std::string planToProtoBytes(const PlanNode& node);
PlanNode    planFromProtoBytes(std::string_view bytes);

std::string scalarExprToProtoBytes(const ScalarExpr& expr);
ScalarExpr  scalarExprFromProtoBytes(std::string_view bytes);

// Protobuf-message-level encoders (no serialization), exposed for callers
// that already have a proto message and want to walk it.
void encodePlan(const PlanNode& node, proto::PlanNode* out);
PlanNode decodePlan(const proto::PlanNode& msg);

void encodeScalarExpr(const ScalarExpr& expr, proto::ScalarExpr* out);
ScalarExpr decodeScalarExpr(const proto::ScalarExpr& msg);

} // namespace quantum::plan
