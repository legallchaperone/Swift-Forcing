// Queue Monitor: a window around the local Python server (qmon/server.py).
// Build with build_app.sh. Needs only the Xcode command line tools.
import Cocoa
import WebKit

let port = ProcessInfo.processInfo.environment["QMON_PORT"] ?? "8765"
let url = URL(string: "http://127.0.0.1:\(port)/")!

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var retries = 0

    func applicationDidFinishLaunching(_ note: Notification) {
        startServer()
        buildMenu()
        let cfg = WKWebViewConfiguration()
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Queue Monitor"
        window.contentView = web
        window.center()
        window.setFrameAutosaveName("QueueMonitorMain")
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { self.web.load(URLRequest(url: url)) }
    }

    func startServer() {
        guard let res = Bundle.main.resourcePath else { return }
        // If a server is already listening (e.g. started by launchd), just use it.
        if let data = try? Data(contentsOf: URL(string: "http://127.0.0.1:\(port)/api/state")!), !data.isEmpty { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        p.arguments = ["-m", "qmon.server"]
        p.currentDirectoryURL = URL(fileURLWithPath: res)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = res
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        p.environment = env
        let logDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/QueueMonitor")
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        let logURL = logDir.appendingPathComponent("server.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let h = try? FileHandle(forWritingTo: logURL) { p.standardOutput = h; p.standardError = h }
        try? p.run()
        server = p
    }

    // Retry until the server is up.
    func webView(_ w: WKWebView, didFailProvisionalNavigation n: WKNavigation!, withError e: Error) {
        retries += 1
        if retries < 60 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { w.load(URLRequest(url: url)) }
        }
    }

    // Open outside links in the default browser.
    func webView(_ w: WKWebView, decidePolicyFor a: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = a.request.url, u.host != "127.0.0.1", a.navigationType == .linkActivated {
            NSWorkspace.shared.open(u); decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Reload", action: #selector(reload), keyEquivalent: "r")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Queue Monitor", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        NSApp.mainMenu = main
    }

    @objc func reload() { web.reload() }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ note: Notification) { server?.terminate() }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
