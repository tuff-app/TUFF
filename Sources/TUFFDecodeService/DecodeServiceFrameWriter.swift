import Darwin
import Foundation
import TUFFDecodeProtocol

/// Serializes complete frames and ensures a stalled helper reader cannot
/// block its model producers indefinitely. A failed or partial frame closes
/// the transport; no later frame is written on a corrupted stream.
final class DecodeServiceFrameWriter: @unchecked Sendable {
    private let handle: FileHandle
    private let timeout: TimeInterval
    private let lock = NSLock()
    private var failed = false

    init(handle: FileHandle, timeout: TimeInterval = 10) {
        self.handle = handle
        self.timeout = max(0.001, timeout)
        DecodeUnixSocket.disableSIGPIPE(on: handle.fileDescriptor)
    }

    func write(_ data: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !failed else { throw POSIXError(.EPIPE) }
        do {
            let fd = handle.fileDescriptor
            var socketType: Int32 = 0
            var socketTypeSize = socklen_t(MemoryLayout<Int32>.size)
            let isSocket = getsockopt(fd, SOL_SOCKET, SO_TYPE, &socketType, &socketTypeSize) == 0
            if !isSocket {
                let flags = fcntl(fd, F_GETFL)
                guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            }
            let deadline = Date().addingTimeInterval(timeout)
            var descriptor = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
            var offset = 0
            while offset < data.count {
                let remaining = deadline.timeIntervalSinceNow
                guard remaining > 0 else { throw POSIXError(.ETIMEDOUT) }
                let result = poll(&descriptor, 1, Int32(min(remaining * 1000, Double(Int32.max))))
                if result < 0, errno == EINTR { continue }
                guard result > 0, descriptor.revents & Int16(POLLOUT) != 0 else {
                    throw POSIXError(result == 0 ? .ETIMEDOUT : .EPIPE)
                }
                let written = data.withUnsafeBytes { bytes in
                    let pointer = bytes.baseAddress!.advanced(by: offset)
                    return isSocket
                        ? Darwin.send(fd, pointer, bytes.count - offset, MSG_DONTWAIT)
                        : Darwin.write(fd, pointer, bytes.count - offset)
                }
                if written > 0 { offset += written; continue }
                if written < 0, errno == EINTR || errno == EAGAIN { continue }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        } catch {
            failed = true
            throw error
        }
    }
}
