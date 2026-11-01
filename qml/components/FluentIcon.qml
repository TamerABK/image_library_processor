import QtQuick
import QtQuick.Window
import "../theme" as Room36Theme

Item {
    id: root

    // Use the artwork drawn for the display size (16, 20, 24 or 48), rather than
    // scaling one master SVG for every control.
    property string name: ""
    property int size: 20
    property string variant: "regular"
    property bool selected: false
    property bool hovered: false
    property color color: theme.black
    property color hoverColor: color
    property color selectedColor: color
    property color disabledColor: theme.disabledText
    readonly property string effectiveVariant: selected ? "filled" : variant
    readonly property color effectiveColor: !enabled ? disabledColor
        : selected ? selectedColor : hovered ? hoverColor : color
    readonly property string colorSuffix: Qt.colorEqual(effectiveColor, theme.white) ? "_white"
        : Qt.colorEqual(effectiveColor, theme.mainBlue) ? "_blue"
        : Qt.colorEqual(effectiveColor, theme.midBlue) ? "_midblue"
        : root.name === "warning" && Qt.colorEqual(effectiveColor, theme.warningRed) ? "_red"
        : root.name === "warning" && Qt.colorEqual(effectiveColor, theme.warningYellow) ? "_yellow" : ""
    readonly property alias status: icon.status
    readonly property bool ready: icon.status === Image.Ready

    Room36Theme.Theme { id: theme }

    implicitWidth: size
    implicitHeight: size

    Image {
        id: icon
        anchors.fill: parent
        source: root.name.length ? theme.fluentIconUrl(root.name + "_" + root.size + "_" + root.effectiveVariant + root.colorSuffix + ".svg") : ""
        sourceSize.width: Math.ceil(width * Screen.devicePixelRatio)
        sourceSize.height: Math.ceil(height * Screen.devicePixelRatio)
        fillMode: Image.PreserveAspectFit
        smooth: true
        opacity: root.enabled ? 1 : 0.36
    }
}
