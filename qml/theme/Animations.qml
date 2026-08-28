import QtQuick

QtObject {
    readonly property int hover: 110
    readonly property int press: 90
    readonly property int selection: 160
    readonly property int popup: 160
    readonly property int panel: 200
    readonly property int page: 220

    readonly property int standardEasing: Easing.OutCubic
    readonly property int emphasizedEasing: Easing.OutQuart
}
