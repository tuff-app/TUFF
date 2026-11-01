import Darwin
import Testing
@testable import TUFFServerCore

@Suite("Server termination signals", .serialized)
struct ServerTerminationSignalTests {
    @Test func dispatchSignalCrossesIntoAsyncCodeWithoutExecutorTrap() async {
        let signals = ServerTerminationSignals([SIGUSR1])
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }

    @Test func repeatedSignalDeliveryKeepsTheFirstSignal() async {
        let signals = ServerTerminationSignals([SIGUSR1])
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)
        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }
    @Test func cancellationBeforeWaitingEndsWithoutSignal() async {
        let signals = ServerTerminationSignals([])
        await signals.cancel()
        #expect(await signals.wait() == nil)
    }

    @Test func cancellationWakesWaiterWithoutSignal() async {
        let signals = ServerTerminationSignals([])
        let waiter = Task { await signals.wait() }
        await Task.yield()
        await signals.cancel()
        #expect(await waiter.value == nil)
    }
}
