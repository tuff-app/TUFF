import Darwin
import Foundation
import TUFFAppCore
import TUFFMacPresentation
import TUFFCLICore
import TUFFServerCore

// The host serves app, HTTP and CLI roles. Inference runs in installed helpers.
if let status = AppWebPDFWorker.runIfRequested(arguments: CommandLine.arguments) {
    exit(status)
}
// `tuff bench` and the Benchmarks screen run the app executable this way.
if let status = AppBenchmarkCommand.runIfRequested(
    arguments: Array(CommandLine.arguments.dropFirst())) {
    exit(status)
}

switch ProcessRole(invokedAs: CommandLine.arguments[0]) {
case .decodeService:
    FileHandle.standardError.write(Data("Inference helpers are installed with each model in TUFF 9.0.\n".utf8))
    exit(1)
case .server:
    Task {
        exit(await RouterServerCommand.main(Array(CommandLine.arguments.dropFirst())))
    }
    dispatchMain()
case .commandLine:
    exit(CLICommand.main(Array(CommandLine.arguments.dropFirst())))
case .app:
    if #available(macOS 26, *) {
        // AppKit's native frame mask keeps the macOS 26 screenshot geometry.
        // This is an undocumented AppKit preference: scope it to this process,
        // before any window is created. Never persist it or change global defaults.
        // Native controls, resizing, glass clipping and shadows remain AppKit-owned.
        let defaults = UserDefaults.standard
        var arguments = defaults.volatileDomain(forName: UserDefaults.argumentDomain)
        arguments["NSConvolutionOverride1"] = Double(AppWindowLayout.windowCornerRadius)
        defaults.setVolatileDomain(arguments, forName: UserDefaults.argumentDomain)
    }
    TUFFMacApp.main()
}
