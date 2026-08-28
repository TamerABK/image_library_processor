import QtQuick

Item {
    visible: false
    width: 0
    height: 0

    Text {
        id: fallbackProbe
        visible: false
    }

    readonly property string fallbackFamily: fallbackProbe.font.family

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

    readonly property bool hasThinFont: thinFont.status === FontLoader.Ready
    readonly property bool hasRegularFont: regularFont.status === FontLoader.Ready
    readonly property bool hasBoldFont: boldFont.status === FontLoader.Ready

    readonly property string thinFamily:
        thinFont.status === FontLoader.Ready ? thinFont.name : fallbackFamily

    readonly property string regularFamily:
        regularFont.status === FontLoader.Ready ? regularFont.name : fallbackFamily

    readonly property string boldFamily:
        boldFont.status === FontLoader.Ready ? boldFont.name : fallbackFamily

    readonly property string displayFamily: thinFamily
    readonly property string pageTitleFamily: boldFamily
    readonly property string sectionTitleFamily: boldFamily
    readonly property string cardTitleFamily: boldFamily
    readonly property string bodyFamily: regularFamily
    readonly property string buttonFamily: boldFamily
    readonly property string captionFamily: regularFamily
    readonly property string detailFamily: regularFamily

    readonly property int display: 48
    readonly property int pageTitle: 34
    readonly property int sectionTitle: 24
    readonly property int cardTitle: 18
    readonly property int body: 16
    readonly property int button: 15
    readonly property int caption: 13
    readonly property int detail: 12

    readonly property int heroSize: pageTitle
    readonly property int titleSize: sectionTitle
    readonly property int bodySize: body
    readonly property int labelSize: caption
    readonly property int detailSize: detail
}
