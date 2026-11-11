import CoreGraphics

public enum AppWindowLayout {
    public static let minimumWidth: CGFloat = 640
    public static let minimumHeight: CGFloat = 440
    public static let defaultWidth: CGFloat = 1_080
    public static let defaultHeight: CGFloat = 720
    // Reference geometry from the macOS 26 product screenshots.
    public static let windowCornerRadius: CGFloat = 28
    public static let sidebarPanelInset: CGFloat = 8.5
    public static let sidebarCornerRadius = windowCornerRadius - sidebarPanelInset
    public static let sidebarMinimumWidth: CGFloat = 196
    public static let sidebarIdealWidth: CGFloat = 196
    public static let sidebarMaximumWidth: CGFloat = 196
    public static let detailMinimumWidth: CGFloat = 420

    public static var minimumHorizontalAllowance: CGFloat {
        sidebarIdealWidth + sidebarPanelInset * 2 + detailMinimumWidth
    }
}
