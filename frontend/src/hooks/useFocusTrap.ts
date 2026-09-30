import { useEffect, useRef, type RefObject } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function focusablesIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => !el.hasAttribute("disabled") && el.getAttribute("aria-hidden") !== "true",
  );
}

// Keeps Tab inside `ref` while `active` and returns focus to whatever opened
// it when the dialog closes. The container itself needs tabIndex={-1} so
// there is always somewhere to land, even before its controls mount.
export function useFocusTrap(
  active: boolean,
  ref: RefObject<HTMLElement | null>,
  initialFocusRef?: RefObject<HTMLElement | null>,
) {
  const restoreRef = useRef<HTMLElement | null>(null);
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    if (!active) return;
    const container = ref.current;
    if (!container) return;
    restoreRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    (initialFocusRef?.current ?? container).focus({ preventScroll: true });

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || !activeRef.current) return;
      const root = ref.current;
      if (!root || !root.contains(document.activeElement)) return;
      const items = focusablesIn(root);
      if (items.length === 0) {
        e.preventDefault();
        root.focus({ preventScroll: true });
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      restoreRef.current?.focus({ preventScroll: true });
      restoreRef.current = null;
    };
  }, [active, ref, initialFocusRef]);
}
