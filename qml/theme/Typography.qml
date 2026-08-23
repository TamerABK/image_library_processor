import QtQuick

Item {
    visible: false
    width: 0
    height: 0

    readonly property string fallbackFamily: Qt.application.font.family

    FontLoader {
        id: thinFont
        source: Qt.resolvedUrl("../../assets/fonts/Britanica Thin.ttf")
    }

    FontLoader {
        id: regularFont
        source: Qt.resolvedUrl("../../assets/fonts/Britanica Regular.ttf")
    }

    FontLoader {
        id: boldFont
        source: Qt.resolvedUrl("../../assets/fonts/Britanica Bold.ttf")
    }

    readonly property bool hasBrandFonts:
        thinFont.status === FontLoader.Ready
        && regularFont.status === FontLoader.Ready
        && boldFont.status === FontLoader.Ready

    readonly property string thinFamily:
        thinFont.status === FontLoader.Ready ? thinFont.name : fallbackFamily

    readonly property string regularFamily:
        regularFont.status === FontLoader.Ready ? regularFont.name : fallbackFamily

    readonly property string boldFamily:
        boldFont.status === FontLoader.Ready ? boldFont.name : fallbackFamily

    readonly property int heroSize: 34
    readonly property int titleSize: 24
    readonly property int bodySize: 16
    readonly property int labelSize: 13
    readonly property int detailSize: 12
}
