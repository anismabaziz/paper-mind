import type { IModelCapabilities } from "@/services/settings";
import { formatTokenCount, formatUsd } from "@/lib/settings-format";

/**
 * What the chosen model can and costs, read before anyone pays for it.
 *
 * The brief needs tool use and structured output, so a reader choosing a model
 * here should be able to see whether it has them rather than find out when a
 * brief is refused. Where the request runs is on the same card, because the two
 * change together: a cheaper model is a model that reads elsewhere.
 */
export default function ModelCapabilities({ model }: { model: IModelCapabilities }) {
  return (
    <div className="border-y border-rule/70 py-3" aria-label="Model capabilities">
      <div className="mb-2 font-mono text-xs text-ink">{model.id}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <Fact label="Context" value={`${formatTokenCount(model.context_window_tokens)} tokens`} />
        <Fact label="Output" value={`${formatTokenCount(model.max_output_tokens)} tokens`} />
        <Fact label="Structured output" value={model.structured_output ? "Supported" : "Not supported"} />
        <Fact label="Tool use" value={model.tool_use ? "Supported" : "Not supported"} />
        <Fact label="Verification timeout" value={`${model.timeout_seconds} seconds`} />
      </div>
      <p className="mt-3 text-xs leading-5 text-ink-soft">
        {formatUsd(model.input_cost_per_million_usd)} input /{" "}
        {formatUsd(model.output_cost_per_million_usd)} output per 1M tokens ·{" "}
        {model.pricing_tier} pricing ·{" "}
        {model.data_location === "cloud" ? "cloud processing" : "local processing"}
      </p>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-ink-faint">{label}</div>
      <div className="mt-0.5 text-ink">{value}</div>
    </div>
  );
}