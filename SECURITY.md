# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in LiteVec, **please report it responsibly**.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Email your report to the maintainers (see the repository owner's GitHub profile for contact info), or use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).
3. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment** within 48 hours of your report.
- **Status update** within 7 days with an assessment of the issue.
- **Fix timeline** depends on severity:
  - **Critical** (data corruption, remote code execution): patch within 72 hours.
  - **High** (denial of service, information disclosure): patch within 2 weeks.
  - **Medium/Low**: included in the next regular release.

### Scope

The following are in scope:

- **litevec-core**: Storage engine, index algorithms, WAL, persistence
- **litevec-ffi**: C FFI boundary (memory safety, buffer overflows)
- **litevec-mcp**: MCP server (input validation, injection attacks)
- **litevec-wasm**: WebAssembly boundary (sandbox escapes)
- **Python/Node bindings**: Type confusion, unsafe deserialization

The following are out of scope:

- Vulnerabilities in dependencies (report upstream)
- Denial of service via legitimate large inputs (expected behavior)
- Issues requiring physical access to the machine

## Security Design

LiteVec is designed with security in mind:

- **Pure Rust core** — memory safety guaranteed by the borrow checker
- **CRC32 checksums** on WAL records detect corruption
- **No network access** — the core library never opens sockets
- **No `unsafe` in public API** — `unsafe` is limited to SIMD intrinsics and FFI boundaries
- **Input validation** — dimension checks, ID sanitization, metadata size limits
